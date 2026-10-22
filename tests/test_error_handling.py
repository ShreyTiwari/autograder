"""What happens when a sheet can't be identified.

A blank AndrewID box, an unreadable one, or a page that isn't a student sheet at
all must never be credited to a student. It has to land in the review queue with
the rest of the row intact, so a human can fix it from the CSV.

No vision model is needed here: the reads a blank box would produce are supplied
directly, which puts the judgment code under test rather than the model.

Every Andrew ID and name in this file is made up; none is a real student.

    python -m unittest discover -s tests -t . -v
"""

import tempfile
import unittest
from pathlib import Path

import fitz

from pipeline import config, ingest, roster, runner
from pipeline.extract import Extraction

ROSTER = {"asmith": "A Smith", "jdoe": "J Doe", "xq": "X Q"}

ZONES = [
    config.Zone("andrew_id", "boxed_id", (0.0, 0.0, 0.5, 0.1), prompt="read the id"),
    config.Zone("response", "freetext", (0.0, 0.2, 1.0, 0.7), prompt="transcribe", points=1),
]
CFG = config.Config(path=Path("test-config.yaml"), zones=ZONES)
FIELDS = config.BASE_FIELDS + config.columns(ZONES) + ["points"]


def build_row(alias, response=""):
    reads = {"andrew_id": Extraction(alias), "response": Extraction(response)}
    return runner._build_row(1, reads, CFG, ROSTER, "2025-09-03")


class TestParticipationScoresOnSubmission(unittest.TestCase):
    """The point follows the identity read, not the transcript.

    Participation is scored on handing a sheet in. Transcribing a hand-drawn
    table is unreliable, so hanging the point on `response` docks a student who
    wrote a full answer the model could not lay out. The printed ID boxes read
    reliably, so that is what the point rides on.
    """

    ZONES = [config.Zone("sheet", "freetext", (0.0, 0.0, 1.0, 1.0),
                         prompt="read the sheet",
                         fields=[config.Field("andrew_id", points=1),
                                 config.Field("response")])]
    CFG = config.Config(path=Path("participation.yaml"), zones=ZONES)

    def row(self, alias, response=""):
        reads = {"andrew_id": Extraction(alias), "response": Extraction(response)}
        return runner._build_row(1, reads, self.CFG, ROSTER, "2025-09-03")

    def test_an_unreadable_response_still_earns_the_point(self):
        row = self.row("jdoe", "")
        self.assertEqual(row["andrew_id"], "jdoe")
        self.assertEqual(row["points"], 1)

    def test_a_transcribed_response_earns_the_same_one_point(self):
        self.assertEqual(self.row("jdoe", "a full answer")["points"], 1)

    def test_an_unsigned_sheet_earns_nothing(self):
        row = self.row("", "a full answer")
        self.assertEqual(row["id_status"], "review")
        self.assertEqual(row["points"], 0)

    def test_the_response_is_still_transcribed_into_the_csv(self):
        self.assertEqual(self.row("jdoe", "kept")["response"], "kept")


class TestUnidentifiedSheet(unittest.TestCase):
    """The ID box is blank or unreadable."""

    def test_blank_read_is_never_credited_to_a_student(self):
        """The roster holds ids as short as 2 characters, so an empty read is
        within MAX_DISTANCE of one. It must not be matched anyway."""
        m = roster.match("", ROSTER)
        self.assertEqual(m.andrew_id, "")
        self.assertEqual(m.status, "review")

    def test_whitespace_read_is_never_credited_to_a_student(self):
        m = roster.match("   ", ROSTER)
        self.assertEqual(m.andrew_id, "")
        self.assertEqual(m.status, "review")

    def test_unreadable_read_goes_to_review(self):
        m = roster.match("xxxxxxxx", ROSTER)
        self.assertEqual(m.andrew_id, "")
        self.assertEqual(m.status, "review")

    def test_a_real_read_still_matches(self):
        """Positive control: review must not be the answer to everything."""
        self.assertEqual(roster.match("asmith", ROSTER).status, "matched")
        self.assertEqual(roster.match("doe", ROSTER).andrew_id, "jdoe")  # fuzzy

    # --- the row that reaches the CSV ---------------------------------------

    def test_unsigned_sheet_still_produces_a_complete_flagged_row(self):
        """The page is not dropped: the work is still transcribed and scored, so
        a human can put a name to it from the CSV instead of hunting for paper."""
        row = build_row("", response="an answer worth a point")
        self.assertEqual(row["andrew_id"], "")
        self.assertEqual(row["id_status"], "review")
        self.assertEqual(row["raw_alias"], "")
        self.assertEqual(row["response"], "an answer worth a point")
        self.assertEqual(row["points"], 1)
        self.assertEqual(set(row), set(FIELDS))

    def test_entirely_blank_page_scores_zero_without_crashing(self):
        row = build_row("", response="")
        self.assertEqual(row["id_status"], "review")
        self.assertEqual(row["points"], 0)
        self.assertEqual(set(row), set(FIELDS))

    def test_validation_accepts_a_review_row(self):
        """An unidentified row is allowed through - flagged, not fatal."""
        runner._validate_output([build_row("")], FIELDS, failed=[])

    def test_validation_accepts_a_dropped_row(self):
        """A page that is not a sheet has no id, and that is not an error."""
        row = build_row("")
        row["id_status"], row["points"] = "dropped", 0
        runner._validate_output([row], FIELDS, failed=[])

    def test_validation_rejects_an_identified_row_with_no_id(self):
        """Guard: the check must actually reject the case it exists for."""
        bad = build_row("asmith")
        bad["andrew_id"], bad["id_status"] = "", "matched"
        with self.assertRaises(AssertionError):
            runner._validate_output([bad], FIELDS, failed=[])


class TestPageFailures(unittest.TestCase):
    """One bad page must not sink the run."""

    def test_failing_page_is_dropped_logged_and_the_rest_survive(self):
        def read(page):
            if page == 2:
                raise RuntimeError("model returned garbage")
            return f"read {page}"

        # the failure must reach run.log, or a short CSV has no explanation
        with self.assertLogs("pipeline", level="ERROR") as captured:
            out = runner._parallel([1, 2, 3], read, key=lambda p: p, stage="test")
        self.assertEqual(set(out), {1, 3}, "a failed page should be dropped, not fatal")
        self.assertIn("page 2", "\n".join(captured.output))


class TestTotalFailureKeepsLastGoodCsv(unittest.TestCase):
    """Every page failing (an expired API key does it) must not destroy the
    results of the last run that worked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.csv = Path(self.tmp.name) / "results.csv"
        self.rows = [build_row("asmith", response="real work")]
        runner._write_csv(self.csv, FIELDS, self.rows)
        self.before = self.csv.read_text()
        self.addCleanup(self.tmp.cleanup)

    def test_a_run_with_no_rows_leaves_the_file_alone(self):
        with self.assertLogs("pipeline", level="ERROR") as captured:
            wrote = runner._write_results(self.csv, FIELDS, [])
        self.assertFalse(wrote, "should report that it wrote nothing")
        self.assertEqual(self.csv.read_text(), self.before, "previous results were lost")
        self.assertIn("untouched", "\n".join(captured.output))

    def test_a_run_with_rows_does_write(self):
        """Guard: the skip must not have turned into 'never write'."""
        more = self.rows + [build_row("jdoe", response="more work")]
        self.assertTrue(runner._write_results(self.csv, FIELDS, more))
        self.assertEqual(len(self.csv.read_text().splitlines()), 3)  # header + 2

    def test_no_csv_is_created_when_there_is_nothing_to_write(self):
        fresh = Path(self.tmp.name) / "never-ran.csv"
        with self.assertLogs("pipeline", level="ERROR"):
            runner._write_results(fresh, FIELDS, [])
        self.assertFalse(fresh.exists(), "an empty run should not leave a stub file")


class TestPagesThatAreNotSheets(unittest.TestCase):
    """Ingest must cope with a page that has no AndrewID header to orient by."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.no_header = cls._pdf("no-header.pdf", "Some other document entirely.")
        cls.blank = cls._pdf("blank.pdf", "")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def _pdf(cls, name, text):
        path = Path(cls.tmp.name) / name
        doc = fitz.open()
        page = doc.new_page()
        if text:
            page.insert_text((72, 200), text, fontsize=14)
        doc.save(path)
        doc.close()
        return path

    def test_page_without_a_header_has_nothing_to_orient_by(self):
        """Establishes why the next two tests matter: the header heuristic
        scores zero, so orientation falls back to Tesseract OSD."""
        with fitz.open(self.no_header) as doc:
            self.assertEqual(ingest._header_score(ingest._render(doc[0])), 0)

    def test_page_without_a_header_still_ingests(self):
        page = ingest.load_page(self.no_header, 1)
        self.assertEqual(page.index, 1)
        self.assertEqual(page.image.mode, "RGB")

    def test_blank_page_still_ingests(self):
        """OSD raises on a page with too few characters; ingest must absorb it."""
        page = ingest.load_page(self.blank, 1)
        self.assertEqual(page.index, 1)
        self.assertEqual(page.image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()

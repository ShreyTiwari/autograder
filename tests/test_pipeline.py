"""One PDF end to end: ingest -> extract -> roster match -> score -> CSV.

Real stage 1 (an actual PDF is rendered and oriented), real caching, real CSV.
Only the model is stubbed: the reads are handed over, so the test pins what the
pipeline does with them. Pages are read in parallel, so the stub hands out its
replies in whatever order they are asked for and every assertion here is by
student, not by page.

Every Andrew ID and name in this file is made up; none is a real student.

    python -m unittest discover -s tests -t . -v
"""

import csv
import io
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import fitz

from pipeline import canvas, config, runner

CONFIG = """
orient:
  region: [0.0, 0.0, 1.0, 0.12]
zones:
  - name: sheet
    kind: freetext
    region: [0.0, 0.0, 1.0, 1.0]
    fields:
      - andrew_id
      - name: response
        points: 1
      - name: q1
        answer: "%s"
        points: 2
    prompt: read the sheet
"""

# The roster carries Canvas's own identity columns, so the upload copies them
# straight across. "zqqq" hands in nothing; "nocanvas" is enrolled here but in
# no Canvas export, so it cannot be uploaded.
ROSTER = """andrew_id,Student,ID,SIS User ID,SIS Login ID,Root Account,Section
jdoe,"Doe, J",101,U-101,jdoe@andrew.cmu.edu,canvas.cmu.edu,17313-1
asmith,"Smith, A",102,U-102,asmith@andrew.cmu.edu,canvas.cmu.edu,17313-1
zqqq,"Qqq, Z",103,U-103,zqqq@andrew.cmu.edu,canvas.cmu.edu,17313-1
nocanvas,"Canvas, No",,,,,
"""

# What the model "reads". Page 2's id is a one-character misread of a roster id.
REPLIES = [
    {"andrew_id": "jdoe", "response": "an answer", "q1": "b"},
    {"andrew_id": "asmth", "response": "", "q1": "c"},
]


def build_tree(root: Path) -> Path:
    """A data/ tree the runner can be pointed at: one 2-page PDF + config + roster."""
    sheets = root / "data" / "participation"
    sheets.mkdir(parents=True)
    (root / "data" / "roster").mkdir(parents=True)
    (root / "data" / "roster" / "roster.csv").write_text(ROSTER)
    (sheets / "config.yaml").write_text(CONFIG % "b")
    pdf = sheets / "2025-09-03.pdf"
    doc = fitz.open()
    for _ in range(len(REPLIES)):
        page = doc.new_page()
        page.insert_text((72, 60), "AndrewID:            Date (MM/DD):", fontsize=14)
    doc.save(pdf)
    doc.close()
    return pdf


@contextmanager
def sandbox(root: Path):
    """Point the pipeline's data/, output/ and logs/ at the temp tree."""
    with mock.patch.multiple(config, DATA=root / "data", OUTPUT=root / "output",
                             LOGS=root / "logs",
                             ROSTER_CSV=root / "data" / "roster" / "roster.csv"):
        yield


def run(root: Path, pdf: Path):
    """Run the pipeline with the model stubbed; return (rows, model calls)."""
    pending, lock = list(REPLIES), threading.Lock()

    def reply(image, prompt, **kw):
        with lock:
            return pending.pop(0)

    # redirect_stderr: the run builds its console handler during the call, so
    # this keeps a whole run's log out of the test output. run.log still has it.
    with sandbox(root), redirect_stderr(io.StringIO()), \
            mock.patch("pipeline.vision.read_json", side_effect=reply) as model:
        out = runner.run(pdf)
    runner.log.handlers.clear()  # they point into the buffer above; don't leak them
    rows = list(csv.DictReader(out.csv.open(newline=""))) if out else []
    return rows, model.call_count


class TestUploadIsHeldUntilEveryPageHasAName(unittest.TestCase):
    """No Canvas file while a sheet is unidentified.

    A student nobody named would be uploaded as absent, which is a wrong grade
    wearing the same face as a right one. The results CSV is still written, so
    the run is not wasted and the rows to fix are visible.
    """

    UNREADABLE = [{"andrew_id": "jdoe", "response": "an answer", "q1": "b"},
                  {"andrew_id": "!!!!!!!!", "response": "an answer", "q1": "b"}]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        with mock.patch.object(sys.modules[__name__], "REPLIES", self.UNREADABLE):
            pdf = build_tree(self.root)
            # no terminal in a test run, so the manual pass leaves it in review
            self.rows, _ = run(self.root, pdf)
        self.out = self.root / "output" / "participation"

    def test_the_results_csv_is_still_written(self):
        self.assertTrue((self.out / "2025-09-03.csv").exists())
        self.assertEqual(len(self.rows), 2)

    def test_the_unreadable_sheet_is_flagged_for_a_human(self):
        statuses = {r["id_status"] for r in self.rows}
        self.assertIn("review", statuses)

    def test_no_canvas_file_is_written(self):
        self.assertFalse((self.out / "2025-09-03-canvas.csv").exists(),
                         "the upload must wait until every sheet has a name")


class TestAnUnsureReadWaitsForAHuman(unittest.TestCase):
    """A scored field read as config.UNSURE is asked of a human in the terminal.

    Unanswered, the row's points stay blank and the Canvas file is held back:
    a guess scored would be a wrong grade that looks right.
    """

    UNSURE = [{"andrew_id": "jdoe", "response": "an answer", "q1": config.UNSURE},
              {"andrew_id": "asmith", "response": "an answer", "q1": "b"}]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        with mock.patch.object(sys.modules[__name__], "REPLIES", self.UNSURE):
            self.pdf = build_tree(self.root)
        self.canvas = self.root / "output" / "participation" / "2025-09-03-canvas.csv"

    def run_with(self, *typed):
        with mock.patch.object(sys.modules[__name__], "REPLIES", self.UNSURE):
            if not typed:  # no terminal in a test run, so nobody is asked
                return run(self.root, self.pdf)[0]
            answers = iter(typed)
            with mock.patch("sys.stdin.isatty", return_value=True), \
                    mock.patch("builtins.input", lambda _: next(answers)), \
                    redirect_stdout(io.StringIO()):
                return run(self.root, self.pdf)[0]

    def test_unanswered_it_has_no_points_and_holds_the_upload(self):
        points = {r["andrew_id"]: r["points"] for r in self.run_with()}
        self.assertEqual(points, {"jdoe": "", "asmith": "3"})
        self.assertFalse(self.canvas.exists())

    def test_answered_it_is_scored_and_the_upload_is_written(self):
        rows = self.run_with("b")
        self.assertEqual({r["andrew_id"]: r["q1"] for r in rows},
                         {"jdoe": "b", "asmith": "b"})
        with self.canvas.open(newline="") as f:
            scores = {r[3].split("@")[0]: r[-1] for r in list(csv.reader(f))[2:]}
        self.assertEqual((scores["jdoe"], scores["asmith"]), ("3", "3"))


class TestWholePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.pdf = build_tree(cls.root)
        cls.rows, cls.calls = run(cls.root, cls.pdf)
        cls.csv_path = cls.root / "output" / "participation" / "2025-09-03.csv"
        cls.csv_text = cls.csv_path.read_text()
        # captured now: a later test re-grades, which rewrites both files
        out = cls.root / "output" / "participation" / "2025-09-03-canvas.csv"
        with out.open(newline="") as f:
            cls.canvas_rows = list(csv.reader(f))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_columns_are_derived_from_the_config(self):
        with self.csv_path.open(newline="") as f:
            header = next(csv.reader(f))
        self.assertEqual(header, config.BASE_FIELDS + ["response", "q1", "points"])

    def test_every_page_becomes_a_row_matched_to_the_roster(self):
        self.assertEqual(len(self.rows), len(REPLIES))
        self.assertEqual({r["date"] for r in self.rows}, {"2025-09-03"})
        self.assertEqual({r["page"] for r in self.rows}, {"1", "2"})
        # "asmth" is one character out, so it still matches; nobody is a duplicate
        self.assertEqual({r["andrew_id"] for r in self.rows}, {"jdoe", "asmith"})
        self.assertEqual({r["id_status"] for r in self.rows}, {"matched"})

    def test_the_answer_key_scores_the_points(self):
        points = {r["andrew_id"]: r["points"] for r in self.rows}
        self.assertEqual(points["jdoe"], "3")    # wrote something (1) + q1 right (2)
        self.assertEqual(points["asmith"], "0")  # wrote nothing, q1 wrong

    def test_the_canvas_upload_has_one_row_per_enrolled_student(self):
        rows = self.canvas_rows
        # the column is named after the run's date; no config lists one
        self.assertEqual(rows[0], config.CANVAS_COLUMNS + ["2025-09-03"])
        # then the Points Possible row Canvas writes, then the students
        self.assertEqual(rows[1][0], canvas.POINTS_POSSIBLE)
        self.assertEqual(rows[1][-1], "1")
        scores = {r[3].split("@")[0]: r[-1] for r in rows[2:]}
        self.assertEqual(scores["jdoe"], "3")       # graded, as in the results CSV
        self.assertEqual(scores["asmith"], "0")
        self.assertEqual(scores["zqqq"], "0")       # no sheet, so the absent score
        # enrolled here but in no Canvas export: nothing to match a row on
        self.assertNotIn("nocanvas", scores)
        self.assertEqual(rows[2][:6], ['Doe, J', '101', 'U-101',
                                       'jdoe@andrew.cmu.edu', 'canvas.cmu.edu',
                                       '17313-1'])

    def test_a_rerun_serves_the_cached_reads(self):
        rows, calls = run(self.root, self.pdf)
        self.assertEqual(calls, 0, "a re-run should not re-read the pages")
        self.assertEqual(self.csv_path.read_text(), self.csv_text)
        self.assertEqual(rows, self.rows)

    def test_editing_the_answer_key_regrades_without_reading_again(self):
        cfg = self.pdf.parent / "config.yaml"
        self.addCleanup(cfg.write_text, CONFIG % "b")
        cfg.write_text(CONFIG % "c")
        rows, calls = run(self.root, self.pdf)
        self.assertEqual(calls, 0, "re-grading must not spend a model call")
        points = {r["andrew_id"]: r["points"] for r in rows}
        self.assertEqual((points["jdoe"], points["asmith"]), ("1", "2"))

    def test_a_filename_with_no_date_is_rejected_before_a_call_is_spent(self):
        undated = self.pdf.parent / "sheets.pdf"
        undated.write_bytes(self.pdf.read_bytes())
        self.addCleanup(undated.unlink)
        with self.assertRaises(ValueError), sandbox(self.root), \
                mock.patch("pipeline.vision.read_json") as model:
            runner.run(undated)
        self.assertEqual(model.call_count, 0)

    def test_a_bad_config_is_rejected_before_a_call_is_spent(self):
        pdf = self.pdf.parent / "2025-09-04.pdf"
        pdf.write_bytes(self.pdf.read_bytes())
        sidecar = pdf.with_suffix(".yaml")
        sidecar.write_text(CONFIG.replace("[0.0, 0.0, 1.0, 1.0]", "[0.5, 0.0, 1.0, 1.0]") % "b")
        self.addCleanup(pdf.unlink)
        self.addCleanup(sidecar.unlink)
        with self.assertRaises(AssertionError), sandbox(self.root), \
                mock.patch("pipeline.vision.read_json") as model:
            runner.run(pdf)
        self.assertEqual(model.call_count, 0)


if __name__ == "__main__":
    unittest.main()

"""The manual pass and the Canvas upload file.

The pipeline end-to-end is covered in test_pipeline; what matters here is what
those two do on their own: that review never hands one student two sheets and
never blocks a run with no terminal, and that the upload file is one row per
enrolled student with Canvas's identity columns copied verbatim.

Every Andrew ID and name in this file is made up; none is a real student.

    python -m unittest discover -s tests -t . -v
"""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from pipeline import canvas, config, review
from pipeline.config import Canvas
from pipeline.extract import Extraction


def ident(aid, cid="1", section="17313-1"):
    return {"Student": f"{aid.title()}, X", "ID": cid, "SIS User ID": f"U-{cid}",
            "SIS Login ID": f"{aid}@andrew.cmu.edu", "Root Account": "canvas.cmu.edu",
            "Section": section}


ROSTER = {"jdoe": ident("jdoe", "101"), "asmith": ident("asmith", "102"),
          "zqqq": ident("zqqq", "103")}


def row(page, alias, aid="", status="review", points=1):
    return {"page": page, "raw_alias": alias, "andrew_id": aid,
            "id_status": status, "response": "something", "points": points}


def resolve(rows, roster, answers):
    """Run the manual pass against a scripted human, with a terminal present."""
    typed = iter(answers)
    with mock.patch("sys.stdin.isatty", return_value=True), \
            mock.patch("builtins.input", lambda _: next(typed)), \
            redirect_stdout(io.StringIO()) as out:
        settled = review.resolve(rows, roster)
    return settled, out.getvalue()


class TestManualReview(unittest.TestCase):

    def test_a_typed_id_settles_a_row_and_is_marked_as_a_human_decision(self):
        rows = [row(1, "jd0e")]
        settled, _ = resolve(rows, ROSTER, ["jdoe"])
        self.assertEqual(settled, 1)
        self.assertEqual(rows[0]["andrew_id"], "jdoe")
        # never promoted to "matched": the CSV should say who decided
        self.assertEqual(rows[0]["id_status"], "manual")

    def test_the_shortlist_can_be_picked_by_number(self):
        rows = [row(1, "jd0e")]
        _, shown = resolve(rows, ROSTER, ["1"])
        self.assertIn("jdoe", shown)
        self.assertEqual(rows[0]["andrew_id"], "jdoe")

    def test_a_student_who_already_has_a_sheet_is_not_on_offer(self):
        rows = [row(1, "jdoe", "jdoe", "matched"), row(2, "jd0e")]
        # "jdoe" is refused, so the human picks someone without a sheet
        settled, shown = resolve(rows, ROSTER, ["jdoe", "zqqq"])
        self.assertIn("not a student without a sheet", shown)
        self.assertEqual(settled, 1)
        self.assertEqual(rows[1]["andrew_id"], "zqqq")
        self.assertFalse(review.duplicates(rows))

    def test_two_rows_cannot_be_given_the_same_student(self):
        rows = [row(1, "jd0e"), row(2, "jdo3")]
        settled, _ = resolve(rows, ROSTER, ["jdoe", "jdoe", "zqqq"])
        self.assertEqual(settled, 2)
        self.assertEqual([r["andrew_id"] for r in rows], ["jdoe", "zqqq"])
        self.assertFalse(review.duplicates(rows))

    def test_a_page_that_is_not_a_sheet_can_be_dropped(self):
        rows = [row(1, "scan artefact", points=1)]
        settled, _ = resolve(rows, ROSTER, ["d"])
        self.assertEqual(settled, 0, "a drop is not an identification")
        self.assertEqual(rows[0]["id_status"], "dropped")
        self.assertEqual(rows[0]["andrew_id"], "")
        self.assertEqual(rows[0]["points"], 0, "a non-sheet earns nothing")

    def test_a_dropped_page_no_longer_holds_up_the_upload(self):
        rows = [row(1, "jdoe", "jdoe", "matched"), row(2, "junk")]
        resolve(rows, ROSTER, ["d"])
        # the gate in runner.run waits on "review" rows only
        self.assertEqual([r for r in rows if r["id_status"] == "review"], [])

    def test_dropping_does_not_use_up_a_student(self):
        rows = [row(1, "jd0e"), row(2, "jd0e")]
        resolve(rows, ROSTER, ["d", "jdoe"])
        self.assertEqual(rows[0]["id_status"], "dropped")
        self.assertEqual(rows[1]["andrew_id"], "jdoe")

    def test_blank_skips_and_q_stops(self):
        rows = [row(1, "jd0e"), row(2, "jdo3"), row(3, "zzz")]
        settled, _ = resolve(rows, ROSTER, ["", "q"])
        self.assertEqual(settled, 0)
        self.assertTrue(all(r["id_status"] == "review" for r in rows))

    def test_a_run_with_no_terminal_skips_instead_of_blocking(self):
        rows = [row(1, "jd0e")]
        with mock.patch("sys.stdin.isatty", return_value=False):
            self.assertEqual(review.resolve(rows, ROSTER), 0)
        self.assertEqual(rows[0]["id_status"], "review")

    def test_end_of_input_leaves_the_rest_in_review(self):
        rows = [row(1, "jd0e"), row(2, "jdo3")]
        with mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch("builtins.input", side_effect=EOFError), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(review.resolve(rows, ROSTER), 0)
        self.assertTrue(all(r["id_status"] == "review" for r in rows))


class TestUnsureReview(unittest.TestCase):

    def reads(self):
        U = config.UNSURE
        return {1: {"andrew_id": Extraction("jdoe"), "q1": Extraction(U),
                    "q2": Extraction(U), "q3": Extraction("c")},
                2: {"andrew_id": Extraction("zqqq"), "q1": Extraction(U)}}

    def ask(self, reads, answers):
        typed = iter(answers)
        with mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch("builtins.input", lambda _: next(typed)), \
                redirect_stdout(io.StringIO()):
            return review.unsure(reads)

    def test_only_unsure_values_are_asked_and_dash_means_nothing(self):
        reads = self.reads()
        self.assertEqual(self.ask(reads, ["B", "-", "a"]), 3)
        self.assertEqual([reads[1]["q1"].value, reads[1]["q2"].value,
                          reads[1]["q3"].value, reads[2]["q1"].value],
                         ["b", "", "c", "a"])

    def test_blank_leaves_it_unsure_and_q_stops(self):
        reads = self.reads()
        self.assertEqual(self.ask(reads, ["", "q"]), 0)
        self.assertEqual(reads[1]["q1"].value, config.UNSURE)
        self.assertEqual(reads[2]["q1"].value, config.UNSURE)

    def test_a_run_with_no_terminal_skips_instead_of_blocking(self):
        reads = self.reads()
        with mock.patch("sys.stdin.isatty", return_value=False):
            self.assertEqual(review.unsure(reads), 0)
        self.assertEqual(reads[1]["q1"].value, config.UNSURE)


class TestDuplicateCheck(unittest.TestCase):

    def test_it_reports_a_student_on_two_sheets(self):
        rows = [row(1, "jdoe", "jdoe", "matched"),
                row(2, "jdoe", "jdoe", "manual")]
        found = review.duplicates(rows)
        self.assertEqual(sorted(found), ["jdoe"])
        self.assertEqual([r["page"] for r in found["jdoe"]], [1, 2])

    def test_unidentified_rows_are_not_duplicates_of_each_other(self):
        self.assertFalse(review.duplicates([row(1, "?"), row(2, "??")]))


class TestUploadFile(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "out-canvas.csv"

    def write(self, rows, roster=None, **kw):
        """Returns (header, points-possible row or None, student rows)."""
        canvas.write(self.path, rows, roster or ROSTER, Canvas(**kw), "2026-09-08")
        with self.path.open(newline="") as f:
            out = list(csv.reader(f))
        if len(out) > 1 and out[1][0] == canvas.POINTS_POSSIBLE:
            return out[0], out[1], out[2:]
        return out[0], None, out[1:]

    def test_the_file_matches_the_shape_canvas_exports(self):
        header, possible, students = self.write([row(1, "jdoe", "jdoe", "matched", 1)])
        self.assertEqual(header, config.CANVAS_COLUMNS + ["2026-09-08"])
        self.assertEqual(possible, [canvas.POINTS_POSSIBLE, "", "", "", "", "", "1"])
        self.assertEqual(students[0], ["Jdoe, X", "101", "U-101",
                                       "jdoe@andrew.cmu.edu", "canvas.cmu.edu",
                                       "17313-1", "1"])

    def test_every_enrolled_student_gets_a_row_including_the_absent(self):
        _, _, students = self.write([row(1, "jdoe", "jdoe", "matched", 1)])
        self.assertEqual(len(students), len(ROSTER))
        self.assertEqual({r[-1] for r in students[1:]}, {"0"})

    def test_an_empty_absent_score_leaves_existing_grades_alone(self):
        _, _, students = self.write([row(1, "jdoe", "jdoe", "matched", 1)], absent="")
        self.assertEqual({r[-1] for r in students[1:]}, {""})

    def test_the_points_possible_row_can_be_turned_off(self):
        _, possible, _ = self.write([], points_possible="")
        self.assertIsNone(possible)

    def test_a_student_with_no_canvas_identity_is_left_out(self):
        roster = dict(ROSTER, nocanvas={c: "" for c in config.CANVAS_COLUMNS})
        _, _, students = self.write([row(1, "jdoe", "jdoe", "matched", 1)],
                                    roster=roster)
        self.assertEqual(len(students), len(ROSTER))
        self.assertNotIn("nocanvas", self.path.read_text())

    def test_an_unidentified_sheet_earns_nobody_anything(self):
        _, _, students = self.write([row(1, "????", "", "review", 1)])
        self.assertEqual({r[-1] for r in students}, {"0"})

    def test_a_quiz_sidecar_can_name_its_own_assignment_with_an_id(self):
        header, _, _ = self.write([], assignment="Quiz 1 (1029812)")
        self.assertEqual(header[-1], "Quiz 1 (1029812)")

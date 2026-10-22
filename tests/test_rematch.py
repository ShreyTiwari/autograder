"""The second pass over unmatched reads.

Pass 1 judges each read against the whole roster and accepts only a close one.
Afterwards one more fact is known - which students already have a sheet - so
the leftovers are retried against just the students who don't. These tests pin
down what that pass may and may not do: it may reach further than pass 1, and
it may break a distance tie on shared characters, but it must never take an id
away from a page that already earned it, and it must never guess.

The reads are modelled on ones the vision model actually produced, but
every Andrew ID and name in this file is made up; none is a real student.

    python -m unittest discover -s tests -t . -v
"""

import unittest

from pipeline import roster, runner

# The four roster entries these reads compete over.
ROSTER = {
    "jmorgan": "Jamie Morgan",
    "rdas": "Rosa Das",
    "mlopez": "Maria Lopez",
    "asmith": "A Smith",
}


def row(page, alias, andrew_id="", status="review"):
    """Just the keys _rematch_unmatched reads and writes."""
    return {"page": page, "raw_alias": alias,
            "andrew_id": andrew_id, "id_status": status}


class TestReachesFurtherThanPassOne(unittest.TestCase):
    def test_pass_one_refuses_these_reads(self):
        """Positive control: without a second pass both are lost."""
        self.assertEqual(roster.match("mwopa2", ROSTER).status, "review")
        self.assertEqual(roster.match("rmoan", ROSTER).status, "review")

    def test_second_pass_resolves_a_read_just_out_of_reach(self):
        """'mwopa2' is 2.5 from 'mlopez' - past MAX_DISTANCE, and the only
        candidate anywhere near it once the matched students are set aside."""
        m = roster.rematch("mwopa2", ROSTER)
        self.assertEqual(m.andrew_id, "mlopez")
        self.assertEqual(m.status, "loose")

    def test_loose_matches_are_labelled_not_silently_promoted(self):
        """A looser match is still a guess a human should confirm, so it must
        be distinguishable from a confident one in the CSV."""
        self.assertNotEqual(roster.rematch("mwopa2", ROSTER).status, "matched")


class TestTieBreak(unittest.TestCase):
    """'rmoan' sits 3 away from both 'jmorgan' and 'rdas'."""

    def test_edit_distance_alone_cannot_separate_them(self):
        self.assertEqual(roster._weighted_distance("rmoan", "jmorgan"),
                         roster._weighted_distance("rmoan", "rdas"))

    def test_shared_characters_break_the_tie(self):
        m = roster.rematch("rmoan", ROSTER)
        self.assertEqual(m.andrew_id, "jmorgan")  # shares r, m, o, a, n
        self.assertEqual(m.status, "loose")

    def test_a_tie_with_no_clear_winner_stays_review(self):
        """Two candidates the same distance away sharing the same characters
        are indistinguishable; picking either would be a coin flip."""
        m = roster.rematch("ab", {"ax": "A X", "ay": "A Y"})
        self.assertEqual(m.andrew_id, "")
        self.assertEqual(m.status, "review")


class TestNeverGuesses(unittest.TestCase):
    def test_a_read_that_is_on_nobody_stays_review(self):
        """'kevinw2' is a correct read of an id that is not on the roster at
        all. Nothing in the leftovers is close, so it must stay for a human."""
        m = roster.rematch("kevinw2", ROSTER)
        self.assertEqual(m.andrew_id, "")
        self.assertEqual(m.status, "review")

    def test_blank_read_is_never_credited_even_with_few_candidates(self):
        """The guard that matters most: with one student left unmatched, an
        empty read is trivially the 'closest' thing to them."""
        m = roster.rematch("", {"rdas": "Rosa Das"})
        self.assertEqual(m.andrew_id, "")
        self.assertEqual(m.status, "review")

    def test_a_short_read_cannot_travel_far(self):
        """3 edits on a 4-character read is not a match, it is a new word."""
        m = roster.rematch("wxyz", {"rdas": "Rosa Das"})
        self.assertEqual(m.status, "review")


class TestDoesNotStealFromMatchedPages(unittest.TestCase):
    def test_an_id_already_claimed_is_off_the_table(self):
        """The whole reason a looser threshold is safe here. Page 1 read
        'jmorgan' cleanly; page 2's bad read must not be given that id."""
        rows = [row(1, "jmorgan", "jmorgan", "matched"), row(2, "rmoan")]
        runner._rematch_unmatched(rows, ROSTER)
        self.assertEqual(rows[1]["andrew_id"], "")
        self.assertEqual(rows[1]["id_status"], "review")

    def test_the_better_fit_wins_a_contested_id(self):
        """Two unmatched reads, both nearest 'jmorgan'. The closer one takes
        it and the other stays for a human - not first-scanned-wins."""
        rows = [row(1, "rmoan"), row(2, "jmorganx")]
        runner._rematch_unmatched(rows, ROSTER)
        self.assertEqual(rows[1]["andrew_id"], "jmorgan")  # 1 edit away
        self.assertEqual(rows[0]["andrew_id"], "")
        self.assertEqual(rows[0]["id_status"], "review")

    def test_matched_rows_are_left_alone(self):
        rows = [row(1, "asmith", "asmith", "matched")]
        runner._rematch_unmatched(rows, ROSTER)
        self.assertEqual(rows[0]["id_status"], "matched")


class TestRankingIsDeterministic(unittest.TestCase):
    def test_a_distance_tie_does_not_depend_on_roster_order(self):
        """Before the tie-break, the winner of a tie was whichever id the
        roster listed first - so the same read could match differently after
        an unrelated roster edit."""
        forward = {"jmorgan": "A D", "rdas": "B K"}
        reverse = {"rdas": "B K", "jmorgan": "A D"}
        self.assertEqual(roster.match("rmoan", forward).distance,
                         roster.match("rmoan", reverse).distance)
        self.assertEqual(roster.rematch("rmoan", forward).andrew_id,
                         roster.rematch("rmoan", reverse).andrew_id)



class TestOneToOne(unittest.TestCase):
    """No student may be credited with two sheets.

    Pass 1 matches each page alone, so two pages can land on the same id - a
    misread of an absent student's id looks exactly like a present one. A
    gradebook summing points by id would double-count it.
    """

    def test_pass_one_alone_does_double_match(self):
        """Positive control: the two reads really do collide."""
        self.assertEqual(roster.match("rdas", ROSTER).andrew_id, "rdas")
        self.assertEqual(roster.match("rdass", ROSTER).andrew_id, "rdas")

    def test_the_closer_read_keeps_the_contested_id(self):
        rows = [row(1, "rdass", "rdas", "matched"), row(2, "rdas", "rdas", "matched")]
        runner._enforce_one_to_one(rows, ROSTER)
        self.assertEqual(rows[1]["andrew_id"], "rdas")  # exact read wins
        self.assertEqual(rows[0]["andrew_id"], "")
        self.assertEqual(rows[0]["id_status"], "review")

    def test_the_loser_is_not_ordering_dependent(self):
        """Same two pages scanned the other way round, same outcome."""
        rows = [row(1, "rdas", "rdas", "matched"), row(2, "rdass", "rdas", "matched")]
        runner._enforce_one_to_one(rows, ROSTER)
        self.assertEqual(rows[0]["andrew_id"], "rdas")
        self.assertEqual(rows[1]["andrew_id"], "")

    def test_a_demoted_row_gets_a_second_chance(self):
        """The point of running enforcement before the second pass: the page
        that lost 'rdas' is a poor read of the student nobody claimed."""
        rows = [row(1, "mwopa2", "rdas", "matched"), row(2, "rdas", "rdas", "matched")]
        runner._enforce_one_to_one(rows, ROSTER)
        runner._rematch_unmatched(rows, ROSTER)
        self.assertEqual(rows[1]["andrew_id"], "rdas")
        self.assertEqual(rows[0]["andrew_id"], "mlopez")  # found its real owner
        self.assertEqual(rows[0]["id_status"], "loose")

    def test_two_identical_reads_leave_one_for_a_human(self):
        """A genuine double submission: one keeps the id, the other is flagged
        rather than silently credited or silently dropped."""
        rows = [row(1, "rdas", "rdas", "matched"), row(2, "rdas", "rdas", "matched")]
        runner._enforce_one_to_one(rows, ROSTER)
        runner._rematch_unmatched(rows, ROSTER)
        ids = [r["andrew_id"] for r in rows if r["andrew_id"]]
        self.assertEqual(ids, ["rdas"])
        self.assertEqual(sum(r["id_status"] == "review" for r in rows), 1)

    def test_unmatched_rows_are_not_treated_as_duplicates(self):
        """Several review rows all share andrew_id "" - that is not a clash."""
        rows = [row(1, "kevinw2"), row(2, "zzzzzz"), row(3, "rdas", "rdas", "matched")]
        runner._enforce_one_to_one(rows, ROSTER)
        self.assertEqual([r["id_status"] for r in rows], ["review", "review", "matched"])

    def test_the_validator_catches_a_duplicate_that_slips_through(self):
        """The backstop: if the two passes above ever regress, the run fails
        loudly instead of writing a CSV that grades one student twice."""
        fields = ["date", "page", "raw_alias", "andrew_id", "id_status", "points"]
        rows = [dict(date="2026-09-03", page=p, raw_alias="rdas",
                     andrew_id="rdas", id_status="matched", points=1)
                for p in (1, 2)]
        with self.assertRaises(AssertionError) as e:
            runner._validate_output(rows, fields, [])
        self.assertIn("same student on two rows", str(e.exception))

if __name__ == "__main__":
    unittest.main()

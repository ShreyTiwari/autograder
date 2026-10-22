"""Roster load + cross-referencing.

The answer set is known: AndrewIDs come from a class roster. So reading an
alias is really "pick the closest roster entry to this noisy signal". This
module loads that source of truth and matches against it, which is what takes
accuracy from ~90% (raw model read) to ~99%.

Correctness lives here:
  - loading asserts the roster is non-empty with unique ids
  - matching returns a Match with a status the runner writes to the CSV, so
    uncertain reads land in a review queue instead of a wrong grade
"""

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from . import config

# Visually confusable pairs cost less than an arbitrary substitution.
CONFUSABLE = {
    frozenset("0o"), frozenset("1l"), frozenset("1i"), frozenset("5s"),
    frozenset("uv"), frozenset("nh"), frozenset("2z"), frozenset("9g"),
}

# Accept a match only if it is this close; otherwise send it to review.
MAX_DISTANCE = 2.0

# Second pass (see rematch): a read that missed everything, retried against
# only the ids no page claimed. Being out this far is allowed there, but never
# more than half the longer string - a 4-character read must not travel 4 -
# and only with a clear winner.
LOOSE_DISTANCE = 4.0
LOOSE_RATIO = 0.5
AMBIGUITY = 1.0  # the runner-up must be at least this much worse
# Edit distance can tie ("rmoan" is 3 from both "jmorgan" and "rdas"). Shared
# characters are a different signal - order-free, so dropped and transposed
# letters don't hurt - and settle a tie when one candidate shares this many
# more of them.
OVERLAP_MARGIN = 2


@dataclass
class Match:
    raw: str
    andrew_id: str  # "" when nothing matched confidently
    distance: float
    status: str  # "matched" | "loose" (second pass, eyeball it) | "review"


def load_roster(path: Path) -> Dict[str, dict]:
    """Return {andrew_id: identity}, where identity holds the Canvas columns.

    The Canvas columns are stored under Canvas's own header names, so the
    gradebook writer copies them across verbatim. Matching only ever looks at
    the keys; the identity is carried for the upload and for showing a human a
    name during review.

    Asserts non-empty, unique ids, and that the Canvas columns are present.
    """
    assert path.exists(), f"roster not found: {path}"
    roster: Dict[str, dict] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in config.CANVAS_COLUMNS if c not in (reader.fieldnames or [])]
        assert not missing, f"{path}: roster is missing column(s) {missing}"
        for row in reader:
            aid = (row.get("andrew_id") or "").strip().lower()
            if not aid:
                continue
            assert aid not in roster, f"duplicate andrew_id in roster: {aid}"
            roster[aid] = {c: (row.get(c) or "").strip() for c in config.CANVAS_COLUMNS}
    assert roster, f"roster is empty: {path}"
    return roster


def name(entry) -> str:
    """The printed name for an identity, for showing a human during review."""
    return entry.get("Student", "") if isinstance(entry, dict) else str(entry)


def suggest(raw: str, candidates: Dict[str, dict], limit: int = 5) -> List[tuple]:
    """The closest candidate ids to a read, as (id, distance), best first.

    The review prompt offers these, so a human picks from a shortlist instead
    of typing an id from memory.
    """
    return [(a, d) for d, _, a in _ranked(raw.strip().lower(), candidates)[:limit]]


def match(raw: str, roster: Dict[str, str]) -> Match:
    """Fuzzy-match a raw read to the closest roster id."""
    raw = raw.strip().lower()
    if not raw:
        # Nothing was read. Ids get as short as 2 characters, so an empty read
        # sits within MAX_DISTANCE of one and would be credited to a real
        # student. An unsigned sheet is a human's problem, not a guess.
        return Match(raw=raw, andrew_id="", distance=float("inf"), status="review")
    best_d, _, best_id = _ranked(raw, roster)[0]
    ok = best_d <= MAX_DISTANCE
    return Match(
        raw=raw,
        andrew_id=best_id if ok else "",
        distance=best_d,
        status="matched" if ok else "review",
    )


def rematch(raw: str, unclaimed: Dict[str, str]) -> Match:
    """Second-pass match, against only the ids no page has claimed yet.

    Pass 1 compares a read to the whole roster and takes only a close one. What
    reaches here is usually a bad read of a student nobody else matched, so the
    candidates are the handful still without a sheet. That is what makes a
    looser ceiling safe: the id of a student who already has a sheet is off the
    table, which is exactly the theft that raising MAX_DISTANCE would invite.

    A small candidate set cuts both ways - noise has less competition - so the
    winner still has to look like an answer: within LOOSE_DISTANCE, no further
    than half the longer string, and clear of the runner-up by AMBIGUITY.
    """
    raw = raw.strip().lower()
    if not raw or not unclaimed:  # an unsigned sheet stays a human's problem
        return Match(raw=raw, andrew_id="", distance=float("inf"), status="review")
    ranked = _ranked(raw, unclaimed)
    best_d, _, best_id = ranked[0]
    ok = (best_d <= LOOSE_DISTANCE
          and best_d <= LOOSE_RATIO * max(len(raw), len(best_id))
          and _is_clear_winner(ranked))
    return Match(
        raw=raw,
        andrew_id=best_id if ok else "",
        distance=best_d,
        status="loose" if ok else "review",
    )


def _is_clear_winner(ranked: List[tuple]) -> bool:
    """True when the top candidate is distinguishable from the next one.

    Either it is AMBIGUITY closer, or - when the two are the same distance
    away, which plain edit distance cannot resolve - it shares OVERLAP_MARGIN
    more characters with the read.
    """
    if len(ranked) == 1:
        return True
    (best_d, best_ov, _), (next_d, next_ov, _) = ranked[0], ranked[1]
    if next_d - best_d >= AMBIGUITY:
        return True
    return next_d == best_d and next_ov - best_ov >= OVERLAP_MARGIN


def _ranked(raw: str, candidates: Dict[str, str]) -> List[tuple]:
    """Candidates as (distance, -shared characters, id), best first.

    Sorting on shared characters as well makes a distance tie deterministic;
    without it the winner is whichever id the roster happened to list first.
    """
    return sorted((_weighted_distance(raw, a), -_overlap(raw, a), a)
                  for a in candidates)


def _overlap(a: str, b: str) -> int:
    """How many characters the two have in common, ignoring order."""
    return sum((Counter(a) & Counter(b)).values())


def _weighted_distance(a: str, b: str) -> float:
    """Edit distance where confusable substitutions cost 0.5 instead of 1."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            if ca == cb:
                sub = prev[j - 1]
            elif frozenset((ca, cb)) in CONFUSABLE:
                sub = prev[j - 1] + 0.5
            else:
                sub = prev[j - 1] + 1
            cur.append(min(sub, prev[j] + 1, cur[j - 1] + 1))
        prev = cur
    return prev[-1]

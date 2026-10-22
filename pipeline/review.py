"""The manual pass: a human names the sheets fuzzy matching could not, and
reads the values the model marked unsure (`unsure`, below).

It runs after both automatic passes, because it needs what they leave behind -
the rows still unidentified, and the ids no page has claimed. Only unclaimed
ids are ever on offer, so an answer given here cannot hand one student two
sheets; the duplicate check afterwards is the proof, not the defence.

**No Canvas file is written while a sheet is still unidentified.** A gradebook
built from a half-identified scan is worse than none: the students it missed go
up as absent, which is a wrong grade wearing the same face as a right one.

A row settled here is `manual`, never promoted to `matched`. The CSV should say
who decided, for the same reason `loose` exists.

Not every page is a sheet. A cover page, a blank, or a scanning artefact can be
`dropped`: it keeps its row, so the CSV still accounts for every page of the
PDF, but it scores nothing and stops holding the upload back. That is different
from skipping, which leaves the page in `review` and keeps the upload waiting.

A run with no terminal skips the pass rather than blocking on input nobody can
give, and leaves those rows in review.
"""

import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import config, roster

log = logging.getLogger("pipeline")

EXCERPT = 90  # characters of transcript shown to jog recognition of the page
SUGGESTIONS = 5

_SKIP, _DROP, _QUIT = object(), object(), object()


def resolve(rows: List[dict], the_roster: Dict[str, dict],
            page_path: Optional[Callable[[int], Path]] = None) -> int:
    """Ask a human to identify the review rows. Returns how many were settled."""
    pending = sorted((r for r in rows if r["id_status"] == "review"),
                     key=lambda r: r["page"])
    if not pending:
        return 0
    if not sys.stdin.isatty():
        log.info("review: %d row(s) need a human, but there is no terminal; "
                 "leaving them for the next interactive run", len(pending))
        return 0

    claimed = {r["andrew_id"] for r in rows if r["andrew_id"]}
    unclaimed = {a: v for a, v in the_roster.items() if a not in claimed}
    print(f"\n{len(pending)} sheet(s) could not be identified. "
          f"{len(unclaimed)} student(s) have no sheet yet.")
    print("No Canvas file is written until every one has a name.")
    print("Enter a number or an AndrewID to name it, d if the page is not a "
          "student's sheet,")
    print("blank to leave it for later, or q to stop.\n")

    resolved = dropped = 0
    for row in pending:
        try:
            choice = _ask(row, unclaimed, page_path)
        except (EOFError, KeyboardInterrupt):
            print()
            choice = _QUIT
        if choice is _QUIT:
            log.info("review: stopped early by the user")
            break
        if choice is _SKIP:
            continue
        if choice is _DROP:
            # Not a sheet: a cover page, a blank, a scanning artefact. It keeps
            # its row so the CSV still accounts for every page of the PDF, but
            # it earns nothing and no longer holds the upload back.
            row["id_status"], row["points"] = "dropped", 0
            dropped += 1
            log.info("review: page %s dropped by hand (read %r); not a sheet",
                     row["page"], row["raw_alias"])
            continue
        row["andrew_id"], row["id_status"] = choice, "manual"
        del unclaimed[choice]
        resolved += 1
        log.info("review: page %s alias=%r -> id=%r (chosen by hand)",
                 row["page"], row["raw_alias"], choice)

    left = sum(r["id_status"] == "review" for r in rows)
    log.info("review: %d of %d rows identified by hand, %d dropped as not a "
             "sheet, %d left in review", resolved, len(pending), dropped, left)
    return resolved


def _ask(row: dict, unclaimed: Dict[str, dict],
         page_path: Optional[Callable[[int], Path]]):
    """Show one unidentified sheet and take an answer for it."""
    print(f"--- page {row['page']}  read as {row['raw_alias']!r}")
    excerpt = " ".join(str(row.get("response") or "").split())[:EXCERPT]
    if excerpt:
        print(f"    wrote: {excerpt}...")
    if page_path:
        print(f"    image: {page_path(row['page'])}")

    options = roster.suggest(row["raw_alias"], unclaimed, SUGGESTIONS)
    for i, (aid, distance) in enumerate(options, 1):
        print(f"    {i}. {aid:<12} {roster.name(unclaimed[aid])}  (off by {distance:g})")

    while True:
        answer = input("    id> ").strip()
        if not answer or answer.lower() == "s":
            return _SKIP
        if answer.lower() == "q":
            return _QUIT
        if answer.lower() == "d":
            return _DROP
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        aid = answer.lower()
        if aid in unclaimed:
            return aid
        print(f"    {aid!r} is not a student without a sheet; pick a number, "
              "type an AndrewID from the list, or d to drop the page")


def unsure(reads: Dict[int, dict],
           page_path: Optional[Callable[[int], Path]] = None) -> int:
    """Ask a human for every value the model read as unsure. Returns how many.

    It runs on the reads, before any judgment: what a human types here is a
    transcription, so scoring treats it like any other read. The answer key is
    never shown, for the same reason. Nothing is saved - the cached reads stay
    the model's - so a re-run asks again, as the identity pass does.
    """
    pending = [(i, name) for i in sorted(reads)
               for name, ex in reads[i].items() if ex.value == config.UNSURE]
    if not pending:
        return 0
    if not sys.stdin.isatty():
        log.info("review: %d unsure value(s) need a human, but there is no "
                 "terminal; leaving them for the next interactive run", len(pending))
        return 0

    print(f"\n{len(pending)} value(s) were too unclear for the model to read.")
    print("No Canvas file is written until every scored one is answered.")
    print("Type what the sheet shows, - if it shows no valid answer, blank to "
          "leave it for later, or q to stop.\n")

    given, shown = 0, None
    for i, name in pending:
        if i != shown:
            print(f"--- page {i}  read as {reads[i]['andrew_id'].value!r}")
            if page_path:
                print(f"    image: {page_path(i)}")
            shown = i
        try:
            answer = input(f"    {name}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            answer = "q"
        if answer.lower() == "q":
            log.info("review: stopped early by the user")
            break
        if not answer:
            continue
        reads[i][name].value = "" if answer == "-" else answer.lower()
        given += 1
        log.info("review: page %d %s: unsure -> %r (given by hand)",
                 i, name, reads[i][name].value)

    log.info("review: %d of %d unsure values given by hand", given, len(pending))
    return given


def duplicates(rows: List[dict]) -> Dict[str, List[dict]]:
    """AndrewIDs claimed by more than one row, checked again after the manual pass.

    Nothing should reach here: the automatic passes enforce one sheet per
    student and review only offers unclaimed ids. It runs anyway because
    double-crediting is the one error a gradebook would carry silently.
    """
    claims: Dict[str, List[dict]] = {}
    for r in rows:
        if r["andrew_id"]:
            claims.setdefault(r["andrew_id"], []).append(r)
    repeated = {a: rs for a, rs in claims.items() if len(rs) > 1}
    for aid, rs in repeated.items():
        log.error("duplicate after review: %r is on pages %s",
                  aid, [r["page"] for r in rs])
    log.info("duplicate check: %d student(s) on more than one sheet", len(repeated))
    return repeated

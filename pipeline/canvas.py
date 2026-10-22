"""The Canvas gradebook upload file.

Canvas resolves a row by the numeric id in its own identity columns, so those
are copied out of the roster untouched rather than rebuilt from a name: no
upload ever falls through to matching students by how they are spelled.

The assignment column is named after the run's date, which the filename already
carries, so nothing here or in any config lists a class date. A column Canvas
does not recognise becomes a new assignment named for that date, which is what
should happen the first time a date is uploaded.

One row per enrolled student, not one per page: Canvas is being told the whole
class's score, so a student with no sheet gets the configured absent score
rather than being left out.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List

from . import config
from .config import Canvas

log = logging.getLogger("pipeline")

POINTS_POSSIBLE = "    Points Possible"  # Canvas writes the row with this indent


def write(path: Path, rows: List[dict], the_roster: Dict[str, dict],
          canvas: Canvas, date: str) -> Path:
    """Write the gradebook upload CSV for one run."""
    column = canvas.column(date)
    scored = {r["andrew_id"]: r["points"] for r in rows if r["andrew_id"]}
    fields = config.CANVAS_COLUMNS + [column]

    out, no_identity = [], []
    for aid, ident in the_roster.items():
        if not ident["ID"]:  # on the roster, in no Canvas export - cannot be uploaded
            no_identity.append(aid)
            continue
        score = scored.get(aid, canvas.absent)
        out.append([ident[c] for c in config.CANVAS_COLUMNS] + [score])
    assert out, "no student on the roster has a Canvas ID; nothing to upload"
    assert all(len(r) == len(fields) for r in out), "canvas row/header mismatch"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        if canvas.points_possible:
            writer.writerow([POINTS_POSSIBLE]
                            + [""] * (len(config.CANVAS_COLUMNS) - 1)
                            + [canvas.points_possible])
        writer.writerows(out)

    graded = len(scored.keys() & the_roster.keys())
    log.info("canvas: %s -> %d students, %d graded, %d at the absent score %r",
             column, len(out), graded, len(out) - graded, canvas.absent)
    # a dropped page is not a sheet, so it is not an unidentified student
    unidentified = sum(1 for r in rows
                       if not r["andrew_id"] and r["id_status"] != "dropped")
    if unidentified:
        log.warning("canvas: %d sheet(s) are still unidentified and earned nothing",
                    unidentified)
    if no_identity:
        log.warning("canvas: %d roster student(s) have no Canvas identity and were "
                    "left out: %s", len(no_identity), ", ".join(no_identity))
    stranded = sorted(set(scored) - set(the_roster))
    if stranded:
        log.warning("canvas: %d graded id(s) are not on the roster: %s",
                    len(stranded), ", ".join(stranded))
    return path

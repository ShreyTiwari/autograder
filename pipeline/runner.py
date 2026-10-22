"""Orchestrates the stages and writes the CSV.

    ingest -> extract (all zones) -> manual read of unsure values
        -> identity via roster -> manual review -> score() -> CSV + Canvas
        upload -> validate

Both stages run in a fixed thread pool (page-level work is independent) and cache
their intermediate results to disk, keyed by PDF content. So a re-run - or a
restart after a crash - skips pages already done and only redoes what's missing.
Failures are never cached, so they retry next run. Cached results are the raw
reads; identity-matching and scoring (the judgment) are re-derived every run, so
transcripts can be re-graded without re-reading images.

Identity handling is common to every assignment: the "andrew_id" zone is read and
roster-matched here. Everything assignment-specific (which zones, how to grade)
comes from the YAML config beside the PDF, so this file never changes between
assignments. output/ and logs/ mirror the data/ tree, so no task name is needed.

Each PDF gets one disposable folder under logs/ (gitignored), holding the
per-page rotation and image caches, the raw-extraction cache, and a run.log of
every read plus any per-page errors, so a human can go eyeball the pages that
need correcting. Only the CSV in output/ is meant to be pushed upstream.
"""

import csv
import hashlib
import json
import logging
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional

from PIL import Image

from . import canvas, config, extract, ingest, review, roster, vision
from .config import Config
from .extract import Extraction

# The date is taken from the filename, not the LLM. Files are named YYYY-MM-DD.
STRICT_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
LOOSE_DATE = re.compile(r"\d{1,4}[-_/.]\d{1,2}[-_/.]\d{1,4}")

log = logging.getLogger("pipeline")


@dataclass
class Results:
    """What a run wrote: the results CSV, and the Canvas upload beside it."""
    csv: Path
    canvas: Optional[Path] = None  # None only when nothing was written at all


def run(pdf_path: Path) -> Optional["Results"]:
    """Process one PDF using the config beside it; write a CSV mirroring data/.

    Returns what was written, or None when every page failed and nothing was.
    """
    cfg = config.load(pdf_path)  # validates before any render or API call
    date = _parse_date(pdf_path)
    the_roster = roster.load_roster(config.ROSTER_CSV)
    rel = _rel(pdf_path)

    cache = _cache_dir(rel, pdf_path, cfg)
    _setup_logging(cache)
    started = perf_counter()
    n = ingest.page_count(pdf_path)
    log.info("run: config=%s pdf=%s date=%s roster=%d pages=%d workers=%d",
             cfg.path, pdf_path, date, len(the_roster), n, config.WORKERS)
    log.info("model: %s/%s", *vision.target())
    log.info("run dir: %s", cache)

    # Stage 1: ingest (parallel, cached per page). Orientation is settled for
    # the whole document first, because the answer for a page whose header is
    # too faint to read lives in the other pages: one stack through one feeder
    # is turned one way, so the majority beats guessing from its handwriting.
    rotations: Dict[int, Optional[int]] = _parallel(
        range(1, n + 1),
        lambda i: _get_rotation(cache, pdf_path, i, cfg.orient, cfg.tokens),
        key=lambda i: i, stage="orient")
    fallback = ingest.majority_rotation(rotations.values())
    decided = {i: (fallback if rotations.get(i) is None else rotations[i])
               for i in range(1, n + 1)}
    unread = [i for i in decided if rotations.get(i) is None]
    log.info("orient: %d/%d pages read a header, majority=%s; %d fell back: %s",
             n - len(unread), n, fallback, len(unread), unread)

    pages: Dict[int, ingest.Page] = _parallel(
        range(1, n + 1),
        lambda i: _get_page(cache, pdf_path, i, cfg.orient, decided[i], cfg.tokens),
        key=lambda i: i, stage="ingest")
    assert pages, f"no pages ingested from {pdf_path}"

    # Stage 2: extract (parallel, cached per page).
    reads: Dict[int, Dict[str, Extraction]] = _parallel(
        pages.values(), lambda p: _get_extractions(cache, cfg, p),
        key=lambda p: p.index, stage="extract")

    # A human reads what the model could not. That is transcription, so it
    # comes before the judgment below, which treats it like any other read.
    review.unsure(reads, lambda i: _page_dir(cache, i) / "page.png")

    # Judgment (cheap, sequential, re-derived each run): roster match + score.
    rows, failed = [], []
    for i in range(1, n + 1):
        if i not in reads:
            failed.append(i)
            continue
        row = _build_row(i, reads[i], cfg, the_roster, date)
        rows.append(row)
        log.info("page %d: alias=%r -> id=%r status=%s",
                 i, row["raw_alias"], row["andrew_id"], row["id_status"])
    _enforce_one_to_one(rows, the_roster)
    _rematch_unmatched(rows, the_roster)

    # The manual pass, then the duplicate check it could have broken. Both need
    # every page judged first, so neither can live inside a per-page function.
    review.resolve(rows, the_roster, lambda i: _page_dir(cache, i) / "page.png")
    review.duplicates(rows)

    fields = config.BASE_FIELDS + config.columns(cfg.zones) + ["points"]
    out_path = config.OUTPUT / rel / f"{pdf_path.stem}.csv"
    wrote = _write_results(out_path, fields, rows)
    upload_path = config.OUTPUT / rel / f"{pdf_path.stem}-canvas.csv"
    unnamed = [r["page"] for r in rows if r["id_status"] == "review"]
    unsure = [r["page"] for r in rows if r["points"] == ""]
    upload = None
    if wrote and (unnamed or unsure):
        # Held back, not written half-done: a student nobody named would go up
        # as absent, a wrong grade wearing the same face as a right one. An
        # older upload is removed with it, or the hold means nothing - the
        # stale file is still there to be uploaded by mistake.
        upload_path.unlink(missing_ok=True)
        if unnamed:
            log.warning("no Canvas file: %d sheet(s) still unidentified, pages %s.",
                        len(unnamed), unnamed)
        if unsure:
            log.warning("no Canvas file: %d sheet(s) still have unsure answers, "
                        "pages %s.", len(unsure), unsure)
        log.warning("Re-run this command from a terminal to settle them; the "
                    "reads are cached, so it costs no model calls.")
    elif wrote:
        upload = canvas.write(upload_path, rows, the_roster, cfg.canvas, date)
    log.info("--- summary ---")
    _log_usage(perf_counter() - started)
    _validate_output(rows, fields, failed)
    log.debug("wrote %s", out_path)  # main.py prints this for the console
    if upload:
        log.debug("wrote %s", upload)
    return Results(csv=out_path, canvas=upload) if wrote else None


# --- stage helpers (cache + parallelism) ------------------------------------

def _parallel(items, fn: Callable, key: Callable, stage: str) -> dict:
    """Run fn over items in a fixed thread pool; keep successes, log failures.

    Returns {key(item): result} for items that succeeded. A failing item is
    logged and dropped (not cached), so a later run retries it.
    """
    out, t0 = {}, perf_counter()
    with ThreadPoolExecutor(max_workers=config.WORKERS) as pool:
        futures = {pool.submit(fn, it): it for it in items}
        for future in as_completed(futures):
            it = futures[future]
            try:
                out[key(it)] = future.result()
            except Exception as e:  # one bad page shouldn't sink the run
                log.error("%s: page %s failed: %s", stage, key(it), e)
                log.debug("%s page %s traceback:\n%s", stage, key(it), traceback.format_exc())
    log.info("%s: %d/%d pages done in %.1fs",
             stage, len(out), len(futures), perf_counter() - t0)
    return out


def _page_dir(cache: Path, index: int) -> Path:
    """A page and all its parts (rotation, full image, zone crops, extractions)
    live together, so a human can open one folder to spot-check a page."""
    return cache / f"page-{index:02d}"


def _get_rotation(cache: Path, pdf_path: Path, index: int, orient: tuple,
                  tokens: tuple) -> Optional[int]:
    """Decide one page's rotation, cached - it costs an OCR sweep per mode.

    None means no mode read the header; the caller settles those against the
    document majority. That is cached too, so a re-run doesn't re-sweep the
    pages that were never going to read.
    """
    path = _page_dir(cache, index) / "rotation.json"
    if path.exists():
        return json.loads(path.read_text())["rotation"]
    rotation = ingest.page_rotation(pdf_path, index, orient, tokens)
    _atomic_save(path, lambda p: p.write_text(json.dumps({"rotation": rotation})))
    return rotation


def _get_page(cache: Path, pdf_path: Path, index: int, orient: tuple,
              rotation: Optional[int], tokens: tuple) -> ingest.Page:
    """Ingest one page at the decided rotation, using the cached image if present."""
    img_path = _page_dir(cache, index) / "page.png"
    if img_path.exists():
        with Image.open(img_path) as im:
            return ingest.Page(index=index, image=im.copy())
    page = ingest.load_page(pdf_path, index, orient, rotation, tokens)
    _atomic_save(img_path, lambda p: page.image.save(p, format="PNG"))
    return page


def _save_crops(page_dir: Path, cfg: Config, page: ingest.Page) -> None:
    """Crop every zone and save it next to the page, so a human can see
    exactly what the vision model is given. Pure CV, no vision model involved -
    doesn't depend on extraction succeeding or being cached."""
    for z in cfg.zones:
        crop = extract.crop_zone(page, z)
        _atomic_save(page_dir / f"{z.name}.png", lambda p, crop=crop: crop.save(p, format="PNG"))


def _get_extractions(cache: Path, cfg: Config, page: ingest.Page) -> Dict[str, Extraction]:
    """Read all zones on a page, using the cached reads if present.

    Writes extractions.json after each zone, not only once the whole page
    succeeds - so if one zone fails, whatever was already read is still on
    disk to spot-check, instead of vanishing with the exception. A cache hit
    requires every value to be present, so a partial file from a failed run
    isn't mistaken for a complete one.
    """
    page_dir = _page_dir(cache, page.index)
    _save_crops(page_dir, cfg, page)
    ex_path = page_dir / "extractions.json"
    expected = {"andrew_id", *config.columns(cfg.zones)}
    if ex_path.exists():
        data = json.loads(ex_path.read_text())
        if set(data) == expected:
            return {name: Extraction(**vals) for name, vals in data.items()}
    reads = {}
    for z in cfg.zones:
        reads.update(extract.extract_zone(page, z))
        payload = json.dumps({name: vars(ex) for name, ex in reads.items()}, indent=2)
        _atomic_save(ex_path, lambda p, payload=payload: p.write_text(payload))
    return reads


def _build_row(index: int, reads: Dict[str, Extraction], cfg: Config,
               the_roster, date: str) -> dict:
    """Identity via the roster, then the config's scoring rules - the judgment
    step, re-derived every run from the cached reads."""
    ident = reads["andrew_id"]  # the loader guarantees this zone exists
    m = roster.match(ident.value, the_roster)
    row = {
        "date": date,
        "page": index,
        "raw_alias": ident.value,
        "andrew_id": m.andrew_id,
        "id_status": m.status,
    }
    points, unsure = 0, []
    for z in cfg.zones:
        for f in z.outputs:
            value = reads[f.name].value.strip()
            if f.name != "andrew_id":  # identity is already in the row
                row[f.name] = value
            if f.points and value == config.UNSURE:
                unsure.append(f.name)
            elif f.points:
                # match the key if there is one, else credit anything written
                ok = value.lower() == f.answer.lower() if f.answer else bool(value)
                points += f.points if ok else 0
    # Blank, not a partial score: a guess would be a wrong grade that looks
    # right. A blank holds the Canvas file back until a human answers it.
    row["points"] = "" if unsure else points
    if unsure:
        log.warning("page %d: unsure read of %s; points left blank for a human",
                    index, ", ".join(unsure))
    return row


def _enforce_one_to_one(rows: List[dict], the_roster) -> None:
    """One sheet per student: the closest claim on a contested id keeps it.

    Pass 1 matches each page on its own, so nothing stops two pages from
    landing on the same student - a misread of an absent student's id can look
    exactly like a present one. Crediting both is the one outcome that must not
    pass silently, because a gradebook summing points by id would double-count
    it. So the closest read keeps the id and the rest go back to review, where
    the second pass can look for the student nobody claimed instead.

    A student who really did hand in two sheets lands here too, and also wants
    a human: which page counts is not ours to decide.
    """
    claims: Dict[str, List[dict]] = {}
    for r in rows:
        if r["andrew_id"]:
            claims.setdefault(r["andrew_id"], []).append(r)
    for aid, contested in claims.items():
        if len(contested) == 1:
            continue
        # distance isn't a column; recompute it for the few rows in question
        contested.sort(key=lambda r: roster.match(r["raw_alias"], the_roster).distance)
        for r in contested[1:]:
            log.warning("duplicate: page %s alias=%r also matched %r, which page %s "
                        "reads more closely; sending page %s to review",
                        r["page"], r["raw_alias"], aid, contested[0]["page"], r["page"])
            r["andrew_id"], r["id_status"] = "", "review"


def _rematch_unmatched(rows: List[dict], the_roster) -> None:
    """Second pass over the review rows, once every page has had its turn.

    Pass 1 judges each read alone against the whole roster. Only afterwards is
    the useful fact available: which students already have a sheet. A read that
    matched nothing is then compared against just the leftovers, where being a
    few characters out is no longer ambiguous (see roster.rematch).

    Rows are taken best-first and claim as they go, so if two of them want the
    same id the better fit wins it and the other stays for a human - rather
    than the id going to whichever page happened to be scanned first.
    """
    claimed = {r["andrew_id"] for r in rows if r["andrew_id"]}
    unclaimed = {a: n for a, n in the_roster.items() if a not in claimed}
    pending = [r for r in rows if r["id_status"] == "review"]
    candidates = []
    for row in pending:
        m = roster.rematch(row["raw_alias"], unclaimed)
        if m.andrew_id:
            candidates.append((m.distance, row, m))
    for _, row, m in sorted(candidates, key=lambda c: c[0]):
        if m.andrew_id not in unclaimed:  # a closer row already took it
            continue
        del unclaimed[m.andrew_id]
        row["andrew_id"], row["id_status"] = m.andrew_id, m.status
        log.info("rematch: page %s alias=%r -> id=%r (distance %.1f, eyeball it)",
                 row["page"], row["raw_alias"], m.andrew_id, m.distance)
    still = sum(r["id_status"] == "review" for r in rows)
    if pending:
        log.info("rematch: %d of %d review rows resolved, %d left",
                 len(pending) - still, len(pending), still)


def _atomic_save(path: Path, write: Callable[[Path], None]) -> None:
    """Write via a temp file + rename so a killed job never leaves a half file
    that a restart would mistake for finished work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write(tmp)
    os.replace(tmp, path)


# --- run setup / IO ---------------------------------------------------------

def _rel(pdf_path: Path) -> Path:
    """output/ and logs/ mirror the data tree: data/quizzes/x.pdf -> quizzes/."""
    try:
        return pdf_path.parent.relative_to(config.DATA)
    except ValueError:
        raise ValueError(f"pdf must live under {config.DATA}: {pdf_path}") from None


def _cache_dir(rel: Path, pdf_path: Path, cfg: Config) -> Path:
    """Stable per-PDF folder for everything disposable (run.log + one subfolder
    per page), keyed by PDF content *and* everything that shaped the reads - so
    editing a region, a prompt, or the orientation settings never serves a read
    taken from the old crop. `answer`/`points` are excluded on purpose:
    re-grading must not force a re-read."""
    pdf = hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:8]
    reads = json.dumps([list(cfg.orient), list(cfg.tokens),
                        [[z.name, z.kind, list(z.region), z.prompt, z.columns]
                         for z in cfg.zones]])
    reads = hashlib.sha1(reads.encode()).hexdigest()[:8]
    d = config.LOGS / rel / f"{pdf_path.stem}-{pdf}-{reads}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _setup_logging(run_dir: Path) -> None:
    """Route logging to this run's run.log (full detail) and the console (clean)."""
    log.setLevel(logging.DEBUG)
    log.handlers.clear()  # fresh handlers per run (avoids dup lines across runs)
    fh = logging.FileHandler(run_dir / "run.log", mode="w")  # fresh each run, not appended
    fh.setLevel(logging.DEBUG)  # full detail incl. tracebacks
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # clean console
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(ch)


def _parse_date(pdf_path: Path) -> str:
    """Pull the class date from the filename (YYYY-MM-DD). No LLM involved.

    Throws if the name has no date, has one in the wrong format, or has one that
    isn't a real calendar date.
    """
    name = pdf_path.name
    m = STRICT_DATE.search(name)
    if m:
        token = m.group()
        try:
            datetime.strptime(token, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"invalid date {token!r} in filename {name!r}: {e}") from e
        return token
    if LOOSE_DATE.search(name):
        raise ValueError(
            f"date in filename {name!r} is not YYYY-MM-DD; rename e.g. 2025-09-03.pdf"
        )
    raise ValueError(
        f"no date in filename {name!r}; expected YYYY-MM-DD, e.g. 2025-09-03.pdf"
    )


def _log_usage(elapsed: float) -> None:
    """What the run cost. Zero calls means every page came from the cache."""
    u = vision.usage()
    log.info("time: %.1fs", elapsed)
    log.info("usage: %d model calls, %s input + %s output tokens",
             u["calls"], f"{u['input']:,}", f"{u['output']:,}")


def _write_results(path: Path, fields: List[str], rows: List[dict]) -> bool:
    """Write the CSV unless the run produced nothing.

    Every page failing is not exotic - an expired API key does it - and writing
    then would truncate the last good results to a bare header. A run that
    achieved nothing has nothing to say.
    """
    if not rows:
        log.error("nothing to write; leaving %s untouched", path)
        return False
    _write_csv(path, fields, rows)
    return True


def _write_csv(path: Path, fields: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_one_line(r) for r in rows)


def _one_line(row: dict) -> dict:
    """Collapse whitespace so one row is one line. A transcript can come back
    with line breaks in it; quoting keeps that valid CSV, but it breaks
    line-based tools and spreadsheet imports. The verbatim read stays in the
    run's extractions.json."""
    return {k: " ".join(v.split()) if isinstance(v, str) else v
            for k, v in row.items()}


def _validate_output(rows: List[dict], fields: List[str], failed: List[int]) -> None:
    """Cheap post-run checks; surface how many rows need a human look."""
    for r in rows:
        assert r["id_status"] in ("matched", "loose", "manual", "review",
                                  "dropped"), r
        assert r["andrew_id"] or r["id_status"] in ("review", "dropped"), r
        assert set(r) == set(fields), \
            f"row/CSV column mismatch: {sorted(set(r) ^ set(fields))}"
    ids = [r["andrew_id"] for r in rows if r["andrew_id"]]
    # 1:1 is built by _enforce_one_to_one + _rematch_unmatched, so a repeat here
    # is a bug in them, not a bad scan - two sheets would be graded as one student.
    assert len(set(ids)) == len(ids), \
        f"same student on two rows: {sorted({i for i in ids if ids.count(i) > 1})}"
    count = lambda s: sum(r["id_status"] == s for r in rows)
    log.info("rows: %d written, %d need id review, %d matched loosely, "
             "%d identified by hand, %d dropped as not a sheet, %d pages failed",
             len(rows), count("review"), count("loose"), count("manual"),
             count("dropped"), len(failed))
    if failed:
        log.warning("failed pages (see run.log): %s", failed)
    if not rows:
        log.error("no rows produced; every page failed - see run.log")

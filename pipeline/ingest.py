"""Stage 1 - Ingest.

PDF -> normalized page images. Pure, deterministic, no vision model. Splits the
PDF into pages and fixes orientation/skew so later stages get clean, upright
crops.

Orientation is the tricky part: the sheets are landscape (AndrewID + Date header
across the top) scanned into portrait pages, so each page is rotated 90 or 270 -
and it can differ per page. We resolve it by OCR'ing the printed header labels
("AndrewID", "Date"): they only read in the upright orientation, so the rotation
where they appear is the right one. Classical OCR here is deterministic and
cacheable - the vision model stays in Stage 2.

We OCR only the header zone (config's `orient` region), not the whole page.
Cropping first is what makes faint scans work: the strip is a couple of printed
labels, so it can be upscaled hard and read with sparse-text mode, instead of
asking one page-wide pass to find small print among a page of handwriting.

A page that still won't read is settled in three escalating steps, cheapest
first:

1. sweep the four rotations with sparse-text OCR (`HEADER_MODES[0]`);
2. sweep them again with a different segmentation mode - a second look reads
   pages the first one loses, and costs nothing on the pages that already read;
3. give up on the page and take `majority_rotation`, how the rest of the
   document was turned.

Only a document where *no* page read a header falls to Tesseract OSD, which
guesses from the handwriting and is what used to hand the vision model sheets
180 degrees out.

Kept separate so its output can be cached and reused across re-runs.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image, ImageOps

from . import config

RENDER_DPI = 200

# Printed header labels live in config (config.HEADER_TOKENS, overridable per
# task with `orient: {tokens: [...]}`), because which words a form prints is a
# property of the form, not of this code. The rotation where they OCR inside
# the header zone is the upright one; this resolves both the 90-vs-270 axis and
# up-vs-down in one shot. Tokens are lenient because OCR mangles the label
# ("date" -> "dato"). No token may contain another: HEADER_MIN_SCORE counts
# distinct tokens as independent evidence, and "date" alongside "dat" would let
# one word score two. config._tokens pins that for a task-supplied list.
HEADER_CONF = 25      # min OCR word confidence to count
HEADER_UPSCALE = 3.0  # autocontrast + upscale rescues faint (pencil) scans
# Two tokens, not one: a single match is as likely to be noise as a label -
# OCR garbage "aaammt" contains "mm", and one such hit used to decide a page.
HEADER_MIN_SCORE = 2
# Tried in order, each a full sweep of the four rotations. psm 11 (sparse text)
# suits a crop of two printed lines; psm 6 (uniform block) reads a different
# subset, so a page the first mode can't see gets a second, differently
# segmented look before it counts as unresolved.
HEADER_MODES = ("--psm 11", "--psm 6")

DESKEW_RANGE = 4.0  # search +/- this many degrees
DESKEW_STEP = 0.5
DESKEW_MIN = 0.3  # don't resample for a negligible tilt


class Page:
    """One scanned sheet, normalized (upright, deskewed)."""

    def __init__(self, index: int, image: Image.Image) -> None:
        self.index = index
        self.image = image  # upright, deskewed RGB PIL image


def page_count(pdf_path: Path) -> int:
    """How many pages the PDF has (so the runner can schedule per-page work)."""
    assert pdf_path.exists(), f"pdf not found: {pdf_path}"
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def page_rotation(pdf_path: Path, index: int,
                  orient: tuple = config.ORIENT_REGION,
                  tokens: tuple = config.HEADER_TOKENS) -> Optional[int]:
    """Which rotation makes page `index` upright, or None if no mode read it.

    Split out from load_page because the answer for an unreadable page lives in
    the other pages: the caller settles those against `majority_rotation` and
    then asks load_page for the image.
    """
    assert pdf_path.exists(), f"pdf not found: {pdf_path}"
    with fitz.open(pdf_path) as doc:
        return _rotation(_render(doc[index - 1]), orient, tokens)


def majority_rotation(rotations: Iterable[Optional[int]]) -> Optional[int]:
    """How most pages were turned, or None if none of them read.

    A scan is one stack of sheets through one feeder, so the odd page whose
    header is too faint to read is almost always turned like its neighbours.
    That is a far better guess than OSD, which reads the handwriting.
    """
    found = [r for r in rotations if r is not None]
    return Counter(found).most_common(1)[0][0] if found else None


def load_page(pdf_path: Path, index: int, orient: tuple = config.ORIENT_REGION,
              rotation: Optional[int] = None,
              tokens: tuple = config.HEADER_TOKENS) -> Page:
    """Render + normalize a single page (1-based). Pure and independent per page.

    `rotation` applies a decision already made (a page's own, or the document
    majority). Without one the page decides for itself and falls to OSD.
    """
    assert pdf_path.exists(), f"pdf not found: {pdf_path}"
    with fitz.open(pdf_path) as doc:
        img = _deskew(_orient(_render(doc[index - 1]), orient, rotation, tokens))
    return Page(index=index, image=img)


def load_pages(pdf_path: Path, orient: tuple = config.ORIENT_REGION,
               tokens: tuple = config.HEADER_TOKENS) -> List[Page]:
    """Convenience: all pages, normalized (sequential). The runner does this in
    parallel with caching; this stays for standalone use."""
    indexes = range(1, page_count(pdf_path) + 1)
    rotations = {i: page_rotation(pdf_path, i, orient, tokens) for i in indexes}
    fallback = majority_rotation(rotations.values())
    return [load_page(pdf_path, i, orient,
                      rotations[i] if rotations[i] is not None else fallback,
                      tokens)
            for i in indexes]


def _render(page) -> Image.Image:
    """Render a PDF page to an RGB image at RENDER_DPI."""
    pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _orient(img: Image.Image, region: tuple = config.ORIENT_REGION,
            rotation: Optional[int] = None,
            tokens: tuple = config.HEADER_TOKENS) -> Image.Image:
    """Rotate to upright, deciding the angle first if one wasn't handed in."""
    if rotation is None:
        rotation = _rotation(img, region, tokens)
    if rotation is None:
        return _osd_orient(img)  # nothing read it; fall back to Tesseract OSD
    return img.rotate(-rotation, expand=True) if rotation else img


def _rotation(img: Image.Image, region: tuple,
              tokens: tuple = config.HEADER_TOKENS) -> Optional[int]:
    """The rotation (degrees clockwise) whose header zone reads, else None.

    Each mode in HEADER_MODES is a full sweep of the four rotations, tried only
    if the one before it settled nothing - so the second look costs nothing on
    a page that already read.
    """
    for mode in HEADER_MODES:
        scores = {a: _header_score(img.rotate(-a, expand=True), region, mode, tokens)
                  for a in (0, 90, 180, 270)}
        best = max(scores, key=lambda a: scores[a])  # ties go to the lowest angle
        if scores[best] >= HEADER_MIN_SCORE:
            return best
    return None


def _header_score(img: Image.Image, region: tuple = config.ORIENT_REGION,
                  mode: str = HEADER_MODES[0],
                  tokens: tuple = config.HEADER_TOKENS) -> int:
    """Count distinct header tokens OCR reads inside the header zone.

    Cropping to the zone before OCR is the point: the crop is a couple of lines
    of print, so psm 11 (sparse text) fits it and a hard upscale is affordable.
    Read page-wide instead, psm 11 has a page of handwriting to get lost in and
    scores 0 everywhere on faint scans - which is what sent them to OSD, and
    OSD then picked a rotation 180 degrees out.
    """
    g = ImageOps.autocontrast(_crop(img, region).convert("L"), cutoff=2)
    g = g.resize((int(g.width * HEADER_UPSCALE), int(g.height * HEADER_UPSCALE)),
                 Image.LANCZOS)
    data = pytesseract.image_to_data(g, config=mode, output_type=pytesseract.Output.DICT)
    found = set()
    for text, conf in zip(data["text"], data["conf"]):
        try:
            if float(conf) < HEADER_CONF:
                continue
        except ValueError:
            continue
        w = "".join(c for c in text.lower() if c.isalnum())
        found.update(t for t in tokens if t in w)
    return len(found)


def _crop(img: Image.Image, region: tuple) -> Image.Image:
    """Crop an [x, y, w, h] page-fraction rectangle."""
    x, y, w, h = region
    return img.crop((int(x * img.width), int(y * img.height),
                     int((x + w) * img.width), int((y + h) * img.height)))


def _osd_orient(img: Image.Image) -> Image.Image:
    """Last resort, when no page in the document read a header: Tesseract's own
    orientation detection. Weak on handwriting - hence everything above it."""
    try:
        osd = pytesseract.image_to_osd(img)
        rot = int(re.search(r"Rotate: (\d+)", osd).group(1))
    except Exception:
        rot = 0
    return img.rotate(-rot, expand=True) if rot else img


def _deskew(img: Image.Image) -> Image.Image:
    """Correct small tilt by maximizing horizontal-projection variance."""
    angle = _skew_angle(np.array(img.convert("L")))
    if abs(angle) < DESKEW_MIN:
        return img
    return img.rotate(angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC)


def _skew_angle(gray: np.ndarray) -> float:
    """Angle (deg) whose rotation makes text lines most horizontal (sharpest rows)."""
    small = Image.fromarray(gray)
    scale = 800 / max(small.size)
    if scale < 1:
        small = small.resize((int(small.width * scale), int(small.height * scale)))
    base = np.asarray(small)
    best_a, best_v = 0.0, -1.0
    for a in np.arange(-DESKEW_RANGE, DESKEW_RANGE + 1e-6, DESKEW_STEP):
        rot = np.asarray(Image.fromarray(base).rotate(a, fillcolor=255))
        v = float((255 - rot.astype(np.float64)).sum(axis=1).var())
        if v > best_v:
            best_v, best_a = v, float(a)
    return best_a

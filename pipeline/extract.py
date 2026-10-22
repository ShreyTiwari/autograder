"""Stage 2 - Extract.

For each zone on a page, crop it and read it with the zone's prompt. Prompts
live in the config, never here, so tuning what the model is asked never means
touching code. Transcription only - no judgment about correctness or grade
(scoring is data in the config, applied by the runner).

One zone is one crop and one model call. It returns one value named after the
zone, or one per key in the zone's `fields`, so two things sitting together on
the page (a written answer and the box circled under it) cost a single call.
"""

from dataclasses import dataclass
from typing import Dict

from . import vision
from .config import Zone
from .ingest import Page


@dataclass
class Extraction:
    value: str


def extract_zone(page: Page, zone: Zone) -> Dict[str, Extraction]:
    """Read one zone; return {column: Extraction} for the values it yields."""
    data = vision.read_json(crop_zone(page, zone), zone.prompt)
    if zone.fields:  # one JSON key per field, named by the prompt
        return {f.name: _clean(zone, f.name, data.get(f.name, ""))
                for f in zone.outputs}
    return {zone.name: _clean(zone, zone.name, data.get("value", ""))}


def _clean(zone: Zone, column: str, raw) -> Extraction:
    """Normalize a raw read for its kind. Identity always gets id normalization:
    a whole-page zone declares one kind, and the id still has no spaces."""
    value = str(raw).strip()
    if column == "andrew_id" or zone.kind == "boxed_id":
        value = "".join(value.split()).lower()  # ids have no spaces
    elif zone.kind == "bubble":
        value = value.lower()  # the prompt defines the vocabulary, not this
    return Extraction(value=value)


def crop_zone(page: Page, zone: Zone):
    """Crop a zone's fractional (x, y, w, h) region out of the page image.

    Public so the runner can save the same crop it sends to the vision model,
    for spot-checking.
    """
    W, H = page.image.size
    x, y, w, h = zone.region
    return page.image.crop((int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)))

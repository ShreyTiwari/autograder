"""Paths, vision settings, and the config loader - the plug-and-play seam.

A task is described by a YAML file that lives next to its PDF, not by Python:
`<pdf-stem>.yaml` if present, else `config.yaml` in the same directory. So one
shared config covers every dated participation sheet, while each quiz - new
layout, new questions, new key - gets its own sidecar. Adding an assignment
means adding a YAML file; the pipeline code never changes.

    zones:
      - name: andrew_id      # required, exactly one: identity, roster-matched
        kind: boxed_id       # boxed_id | freetext | bubble (how to clean a read)
        region: [0.08, 0.075, 0.54, 0.075]   # x, y, w, h as page fractions
        prompt: "..."        # required: every prompt lives here, none in code
      - name: q1
        kind: bubble
        region: [0.05, 0.195, 0.90, 0.205]
        prompt: "..."
        answer: "c"          # optional: ground truth
        points: 1
      - name: sheet          # one crop, one call, every value on the page
        kind: freetext
        region: [0.0, 0.0, 1.0, 1.0]
        fields:              # JSON keys the prompt asks for; one column each
          - andrew_id        # a bare name transcribes only
          - name: response   # a mapping scores that field on its own
            points: 1
        prompt: "..."

A zone is one crop read by one prompt. It yields one CSV column named after the
zone, or - with `fields` - several, one per key the prompt asks the model for,
so related things on one part of the page cost a single call.

Scoring is data too, and it attaches per field: points 0 (the default)
transcribes without scoring; points > 0 scores against `answer` if one is given,
else on the value being non-empty. A single-value zone carries `answer`/`points`
itself; a `fields` zone carries them on each field. Reading stays separate from judgment - `answer`/`points` are
re-applied every run, never baked into the cached reads.

The schema is validated at load, which is the first thing a run does, so a bad
region fails before a page renders or an API call is spent.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
LOGS = ROOT / "logs"  # per-run debug trail (gitignored); only OUTPUT is pushed
ROSTER_CSV = DATA / "roster" / "roster.csv"

CONFIG_NAME = "config.yaml"  # shared fallback when a PDF has no sidecar

# Where the printed header labels sit on an *upright* sheet, as page fractions.
# Ingest OCRs only this zone at each of the four rotations; the one where the
# labels read is the upright one. It lives here, and can be overridden per task
# with a top-level `orient: {region: [x, y, w, h]}`, so a form whose labels sit
# elsewhere is a config change rather than a code change.
ORIENT_REGION = (0.0, 0.0, 1.0, 0.32)

# The printed words OCR looks for in that zone. The default suits the
# participation sheet ("AndrewID:", "Date (MM/DD):"); a form printed differently
# overrides it with `orient: {tokens: [...]}` rather than editing ingest. A quiz
# with no date field, for instance, can never match two of these.
HEADER_TOKENS = ("andrew", "dat", "mm", "dd")

# What a prompt tells the model to return for a value it can't read with
# confidence. A scored field read this way leaves the row's points blank for a
# human, instead of scoring a guess.
UNSURE = "?"

# Columns every task shares; zone names + "points" are appended per config.
BASE_FIELDS = ["date", "page", "raw_alias", "andrew_id", "id_status"]
RESERVED = set(BASE_FIELDS) | {"points"}

# Canvas gradebook identity columns, in the order Canvas writes and reads them.
# The roster stores them under these exact names so the upload is a straight
# copy. See README for how to refresh them from a Canvas export.
CANVAS_COLUMNS = ["Student", "ID", "SIS User ID", "SIS Login ID", "Root Account",
                  "Section"]

# Canvas ignores a column named like one of its own, and silently refuses to
# match an assignment whose name contains any of these.
CANVAS_RESERVED = {c.lower() for c in CANVAS_COLUMNS} | {"integration id"}
CANVAS_BANNED = ("current score", "current points", "current grade",
                 "final score", "final points", "final grade",
                 "override score", "override grade", "override status")

# Canvas writes an assignment column as "Title (12345)" and resolves the id
# first, so a column carrying one survives the assignment being renamed.
ASSIGNMENT_HEADER = re.compile(r"^(?P<title>.*?)\s*(?:\((?P<id>\d+)\))?$")

# The assignment column is named after the run's date, so no config anywhere
# lists a date. A task whose column needs another shape overrides the template;
# {date} is substituted with the run's YYYY-MM-DD.
ASSIGNMENT = "{date}"
DATE_FIELD = "{date}"

CANVAS_KEYS = {"assignment", "points_possible", "absent"}

ZONE_KEYS = {"name", "kind", "region", "prompt", "fields", "answer", "points"}
FIELD_KEYS = {"name", "answer", "points"}  # a field is scored just like a zone
KINDS = ("boxed_id", "freetext", "bubble")  # how a read is cleaned up

# Vision model for Stage 2. Provider is pluggable; only the chosen provider's SDK
# and API key are needed. Override at runtime with VISION_PROVIDER / VISION_MODEL.
# Use the current flagship of each provider. NOTE: only the Anthropic id is
# verified here; confirm the openai/gemini ids against each provider's latest
# lineup (model names move fast) - each is a one-line change.
VISION_PROVIDER = "anthropic"  # anthropic | openai | gemini | openrouter | lmstudio | bedrock
VISION_MODEL = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5.6",           # verify: current GPT-5.6-series vision model id
    "gemini": "gemini-3.8-flash",  # verified available
    # openrouter ids are "<vendor>/<model>", so any provider's model is one
    # VISION_MODEL away. Verified: `curl https://openrouter.ai/api/v1/models`.
    # Default is a free open-weights vision model; the ":free" tier is rate
    # limited, so expect 429s that read_image's backoff absorbs.
    "openrouter": "minimax/minimax-m3:free",
    "lmstudio": "qwen/qwen3.8-27b",  # verified: `curl $LMSTUDIO_BASE_URL/models`
    "bedrock": "qwen.qwen3-vl-235b-a22b",
}

# openrouter proxies every vendor behind one OpenAI-compatible endpoint and one
# key (OPENROUTER_API_KEY), so switching models needs no new SDK.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# lmstudio talks to a local server instead of a cloud API - no key needed.
# Use 127.0.0.1, not localhost - on this machine localhost resolves to a
# different address than the one LM Studio's server actually listens on.
LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
# Local inference is slow, and concurrent requests share one GPU - a whole-page
# read under load can take many minutes, well past the SDK's 600s default.
LMSTUDIO_TIMEOUT = 1800.0  # seconds per request

# Fixed parallelism for both stages (page-level work is independent).
WORKERS = 8


@dataclass
class Field:
    """One JSON key the prompt asks for: a CSV column, scored on its own."""
    name: str
    answer: Optional[str] = None  # ground truth; matched case-insensitively
    points: int = 0  # 0 = transcribe only


@dataclass
class Zone:
    """A region on the page, how to read it, and how (if at all) to score it."""
    name: str
    kind: str  # "boxed_id" | "freetext" | "bubble"
    region: Tuple[float, float, float, float]  # x, y, w, h as page fractions
    prompt: str  # asked of the vision model; prompts live only in config
    fields: Optional[List[Field]] = None  # JSON keys; None = one value, the zone
    answer: Optional[str] = None  # single-value zone: ground truth
    points: int = 0  # single-value zone: 0 = transcribe only

    @property
    def outputs(self) -> List[Field]:
        """Every value this zone yields, each with its own scoring rule."""
        return self.fields or [Field(self.name, self.answer, self.points)]

    @property
    def columns(self) -> List[str]:
        """The CSV columns this zone produces."""
        return [f.name for f in self.outputs]


@dataclass
class Canvas:
    """Where a run's points go in the Canvas gradebook.

    The column is named after the run's date, which the filename already
    carries, so one shared config serves every class date and no file anywhere
    lists one. A task that needs another shape sets `assignment`, where {date}
    stands for the run's date: a quiz sidecar covers a single PDF, so it can
    name its Canvas assignment outright and include the id.
    """
    assignment: str = ASSIGNMENT  # column header template; {date} is substituted
    points_possible: str = "1"  # the Points Possible row; "" leaves the row out
    absent: str = "0"  # score for a student with no sheet; "" leaves their grade alone

    def column(self, date: str) -> str:
        """The Canvas column header for a run."""
        return self.assignment.replace(DATE_FIELD, date)


@dataclass
class Config:
    path: Path  # the file this came from; logged so a run says what it used
    zones: List[Zone]
    orient: Tuple[float, float, float, float] = ORIENT_REGION  # header zone
    tokens: Tuple[str, ...] = HEADER_TOKENS  # printed labels that zone must show
    canvas: Canvas = field(default_factory=Canvas)  # gradebook upload, if configured


def columns(zones: List[Zone]) -> List[str]:
    """Every column the zones produce; andrew_id is identity, not a column."""
    return [c for z in zones for c in z.columns if c != "andrew_id"]


def load(pdf_path: Path) -> Config:
    """Load the config beside a PDF: its <stem>.yaml sidecar, else config.yaml."""
    sidecar = pdf_path.with_suffix(".yaml")
    shared = pdf_path.parent / CONFIG_NAME
    path = sidecar if sidecar.exists() else shared
    if not path.exists():
        raise FileNotFoundError(
            f"no config for {pdf_path}; expected {sidecar} or {shared}")
    data = yaml.safe_load(path.read_text()) or {}
    zones = [_zone(z, path) for z in data.get("zones") or []]
    _validate(zones, path)
    orient = data.get("orient") or {}
    assert isinstance(orient, dict), f"{path}: orient must be a mapping, got {orient!r}"
    unknown = set(orient) - {"region", "tokens"}
    assert not unknown, f"{path}: unknown orient keys {sorted(unknown)}"
    region = (_region(orient["region"], "orient", path) if orient.get("region")
              else ORIENT_REGION)
    return Config(path=path, zones=zones, orient=region,
                  tokens=_tokens(orient.get("tokens"), path),
                  canvas=_canvas(data.get("canvas"), path))


def _canvas(value, path: Path) -> Canvas:
    """Validate the Canvas upload block, or take the defaults.

    Checked here, at load, because a name Canvas refuses fails silently at
    upload time - the column is simply ignored and nobody is graded.
    """
    if value is None:
        return Canvas()
    assert isinstance(value, dict), f"{path}: canvas must be a mapping, got {value!r}"
    unknown = set(value) - CANVAS_KEYS
    assert not unknown, f"{path}: unknown canvas keys {sorted(unknown)}"
    return Canvas(assignment=_assignment(value.get("assignment"), path),
                  points_possible=_number(value.get("points_possible")),
                  absent=_number(value.get("absent"), default="0"))


def _assignment(header, path: Path) -> str:
    """Check a "Title (12345)" column header against Canvas's own rules.

    The date is not substituted yet, so what is checked is the fixed part of
    the name - which is where a banned word would be written anyway.
    """
    if header is None:
        return ASSIGNMENT
    assert isinstance(header, str) and header.strip(), \
        f"{path}: canvas assignment must be a non-empty column header"
    header = header.strip()
    title = (ASSIGNMENT_HEADER.match(header).group("title") or "").strip()
    assert title, f"{path}: canvas assignment {header!r} has no title"
    assert title.lower() not in CANVAS_RESERVED, \
        f"{path}: canvas assignment {title!r} is a reserved Canvas column name"
    for banned in CANVAS_BANNED:
        assert banned not in title.lower(), \
            f"{path}: canvas assignment {title!r} contains {banned!r}, which Canvas refuses"
    return header


def _number(value, default: str = "") -> str:
    """A score from YAML as the string the CSV carries; None means the default."""
    if value is None:
        return default
    assert isinstance(value, (int, float, str)), \
        f"canvas score {value!r} must be a number or a string"
    return str(value).strip()


def _tokens(value, path: Path) -> Tuple[str, ...]:
    """Validate the printed labels OCR should look for, or take the default.

    Two must match before a rotation is believed, so there have to be at least
    two - and because they are counted as independent evidence, none may
    contain another ("date" alongside "dat" lets one word score twice).
    """
    if value is None:
        return HEADER_TOKENS
    assert isinstance(value, list) and len(value) >= 2, \
        f"{path}: orient tokens must be a list of at least 2, got {value!r}"
    for t in value:
        # tokens are matched against OCR text stripped to lowercase alphanumerics
        assert isinstance(t, str) and t.isalnum() and t.islower(), \
            f"{path}: orient token {t!r} must be a lowercase alphanumeric string"
    for i, a in enumerate(value):
        for j, b in enumerate(value):
            assert i == j or a not in b, \
                f"{path}: orient token {a!r} is inside {b!r}; they must be independent"
    return tuple(value)


def _zone(d: dict, path: Path) -> Zone:
    """Build one Zone, checking what a hand-edited file gets wrong."""
    assert isinstance(d, dict), f"{path}: each zone must be a mapping, got {d!r}"
    unknown = set(d) - ZONE_KEYS
    assert not unknown, f"{path}: unknown zone keys {sorted(unknown)}"
    for key in ("name", "kind", "region", "prompt"):
        assert d.get(key), f"{path}: zone {d.get('name', d)!r} is missing {key!r}"
    assert d["kind"] in KINDS, \
        f"{path}: zone {d['name']!r} has unknown kind {d['kind']!r}; known: {list(KINDS)}"

    raw_fields = d.get("fields")
    fields = None
    if raw_fields is not None:
        assert isinstance(raw_fields, list) and raw_fields, \
            f"{path}: zone {d['name']!r} fields must be a non-empty list"
        fields = [_field(f, d["name"], path) for f in raw_fields]
        names = [f.name for f in fields]
        assert len(set(names)) == len(names), \
            f"{path}: zone {d['name']!r} repeats a field name"
        # each field carries its own scoring, so zone-level scoring is ambiguous
        assert not d.get("answer") and not d.get("points"), \
            f"{path}: zone {d['name']!r} has fields, so score each field, not the zone"

    region = _region(d["region"], f"zone {d['name']!r}", path)

    answer = d.get("answer")
    # YAML 1.1: a bare n/y/no/on arrives as a bool, silently unmatchable.
    assert answer is None or isinstance(answer, str), \
        f"{path}: zone {d['name']!r} answer {answer!r} must be quoted"

    return Zone(name=d["name"], kind=d["kind"], region=region,
                prompt=d["prompt"], fields=fields, answer=answer,
                points=int(d.get("points", 0)))


def _region(value, what: str, path: Path) -> Tuple[float, float, float, float]:
    """Validate an [x, y, w, h] page-fraction rectangle."""
    assert value is not None and len(value) == 4, \
        f"{path}: {what} region needs 4 numbers, got {value!r}"
    x, y, w, h = (float(v) for v in value)
    assert 0 <= x and 0 <= y and w > 0 and h > 0 and x + w <= 1 and y + h <= 1, \
        f"{path}: {what} region {value} runs off the page"
    return x, y, w, h


def _field(f, zone_name: str, path: Path) -> Field:
    """One JSON key the prompt asks for: a bare name, or a name plus scoring."""
    if isinstance(f, str):
        f = {"name": f}
    assert isinstance(f, dict), \
        f"{path}: zone {zone_name!r} field {f!r} must be a name or a mapping"
    unknown = set(f) - FIELD_KEYS
    assert not unknown, \
        f"{path}: zone {zone_name!r} field has unknown keys {sorted(unknown)}"
    name = f.get("name")
    assert isinstance(name, str) and name, \
        f"{path}: zone {zone_name!r} has a field with no name"
    answer = f.get("answer")
    # YAML 1.1: a bare n/y/no/on arrives as a bool, silently unmatchable.
    assert answer is None or isinstance(answer, str), \
        f"{path}: field {name!r} answer {answer!r} must be quoted"
    return Field(name=name, answer=answer, points=int(f.get("points", 0)))


def _validate(zones: List[Zone], path: Path) -> None:
    assert zones, f"{path}: no zones defined"
    names = [z.name for z in zones]
    assert len(set(names)) == len(names), f"{path}: duplicate zone names in {names}"
    idents = [z.name for z in zones if "andrew_id" in z.columns]
    assert len(idents) == 1, \
        f"{path}: needs exactly one zone producing 'andrew_id' (identity), got {names}"
    for z in zones:
        for f in z.outputs:
            assert f.points >= 0, f"{path}: {f.name!r} has negative points"
            assert not (f.answer and not f.points), \
                f"{path}: {f.name!r} has an answer but points: 0, so it is never scored"
    cols = columns(zones)  # andrew_id's read becomes raw_alias, not a column
    assert len(set(cols)) == len(cols), f"{path}: duplicate column names in {cols}"
    for c in cols:
        assert c not in RESERVED, f"{path}: column {c!r} collides with a base CSV column"

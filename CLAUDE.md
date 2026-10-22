# 17-313 Scan Pipeline

Extract structured data from scanned PDFs of student sheets. First use case:
pull each student's AndrewID alias off participation sheets into a CSV.
Designed to be reused for quiz grading by changing config, not code.

## Design principles (from the user)

- **DO NOT OVER-ENGINEER.** This is the top rule. Clean simple code = best code.
  Make the most minimal set of changes needed. No speculative abstraction, no
  features nobody asked for.
- **Comments: minimal, concise, to the point.** Only comment what the code can't
  say itself. No narration, no restating the obvious.
- The pipeline must switch between **participation** and **quiz** scanning via
  **config, not new code**.
- **Correctness matters**: cross-reference against the roster, put assertion
  checks in the code, and make outputs easy to spot-check.
- Keep **transcription separate from judgment** — read the sheet first, decide
  correctness/grade second, so a bad read and a bad grade are distinguishable
  and transcripts can be re-graded without re-reading images.
- Don't ask the vision model to self-verify its own read (e.g. counting boxes
  in a crop it just read). Models are unreliable at counting small repeated
  elements in both directions — it can flag a good crop as bad or miss an
  actually bad one — so it's not a real check. A crop-sanity check needs to be
  deterministic (geometry/image-processing), not model-asked.

## Layout

```
data/
  participation/config.yaml  shared config: every participation PDF uses it
  participation/<date>.pdf   scanned sheets, one PDF per class, named by date
  quizzes/<date>-<name>.pdf  quiz scans
  quizzes/<date>-<name>.yaml sidecar config: that quiz only (zones + key)
  roster/roster.csv          source of truth (91 students): andrew_id, then
                             Canvas's own identity columns spelled as Canvas
                             spells them (Student, ID, SIS User ID, SIS Login
                             ID, Root Account, Section), so the upload copies
                             them verbatim. Rebuilt from a Canvas gradebook
                             export, joined on the local part of SIS Login ID
  badging/                   future: badge-swipe attendance to reconcile
output/
  <mirrors data/>/<stem>.csv        extracted results
  <mirrors data/>/<stem>-canvas.csv the Canvas gradebook upload
pipeline/
  config.py    paths, vision settings, the YAML config loader (the swap point)
  ingest.py    Stage 1: PDF -> deskewed page images (pure CV, cacheable)
  extract.py   Stage 2: crop a zone -> vision model -> value(s)
  roster.py    load roster, fuzzy-match reads, cross-reference (assertions)
  review.py    the manual pass: a human names what matching could not
  canvas.py    the gradebook upload file
  runner.py    orchestrate stages -> write CSV -> validate
main.py        CLI: python main.py <pdf_path>
```

## Pipeline stages

1. **Ingest** — PDF to normalized (upright, deskewed) pages. Deterministic.
   Orientation OCRs the config's `orient` region (the printed header labels) at
   all four rotations; the one that reads is upright. Cropping to that zone
   before OCR is what makes faint scans work. A page that still won't read gets
   a second sweep in another OCR mode, then the rotation most of the document
   got (`runner`'s orient pass) — see below.
2. **Extract** — per-zone crop read with the zone's prompt from the config;
   returns the raw value(s). Transcription only — no grade decision here.
3. **Cross-reference** — fuzzy-match the read against the roster; confusable
   pairs (0/o, 1/l, 5/s...) cost less. Confident unique match is accepted;
   otherwise `status=review`. Then a **second pass** (`runner._rematch_unmatched`)
   retries the review rows against only the ids no page claimed — see below.
4. **Review** — the run stops and asks a human to name the sheets neither pass
   could (`review.resolve`), offering only students who have no sheet yet. Then
   `review.duplicates` re-checks that nobody ended up on two — see below.
5. **Score** — the config's per-field `answer`/`points` turn the parsed JSON
   into a `points` column. Declarative, re-derived every run.
6. **Upload** — `canvas.write` turns those points into a Canvas gradebook CSV:
   one row per enrolled student, identity copied from the roster — see below.
7. **Validate** — post-run checks; report how many rows need a human look.

## Config contract (plug-and-play)

A task is described by a **YAML file next to its PDF**, never by Python. The
runner resolves it one way: `<pdf-stem>.yaml` beside the PDF if it exists, else
`config.yaml` in the same directory. So participation keeps one shared config
for every class date, and each quiz — new layout, new questions, new key — gets
its own sidecar. `output/` and `logs/` mirror the `data/` subtree, so the CLI
takes a PDF path and nothing else.

```yaml
orient:                    # optional: where the printed header labels sit
  region: [0.0, 0.0, 1.0, 0.32]   # defaults to config.ORIENT_REGION
  tokens: ["andrew", "dat"]       # defaults to config.HEADER_TOKENS
zones:
  - name: andrew_id        # required, exactly one: identity, roster-matched
    kind: boxed_id         # boxed_id | freetext | bubble (how a read is cleaned)
    region: [x, y, w, h]   # page fractions, origin top-left
    prompt: "..."          # required: every prompt lives in config, none in code
  - name: q1
    kind: bubble
    region: [...]
    prompt: "..."
    answer: "c"            # optional ground truth; quote it (YAML turns no/yes into bools)
    points: 1
  - name: sheet            # one crop, one call, every value on the page
    kind: freetext
    region: [...]
    fields:                # JSON keys the prompt asks for; one CSV column each
      - andrew_id          # a bare name transcribes only
      - name: q3           # a mapping scores that field on its own
        answer: "b"
        points: 1
    prompt: "..."
canvas:                    # optional: everything has a working default
  assignment: "{date}"     # the Canvas column header; {date} is the run's date
  points_possible: 1       # optional: the Points Possible row, 1 by default
  absent: "0"              # optional: score for a student with no sheet
```

Rules:
- **Prompts live only in config.** `extract.py` holds no prompt text; tuning what
  the model is asked never touches code.
- A zone is one crop and one model call. It yields one column named after the
  zone, or one per name in `fields` — so things sitting together on the page cost
  a single call.
- **Scoring is data, per field**: `points: 0` (default) transcribes without
  scoring; `points: N` with an `answer` scores on a case-insensitive match;
  `points: N` without one scores on the field being non-empty. `points` is the
  auto-scored total only — open responses stay hand-graded from their transcript
  column.
- Scoring attaches to whatever the prompt asks for: a single-value zone carries
  `answer`/`points` itself, and a `fields` zone carries them per field (so a
  zone with `fields` must not also set them at zone level).
- **Unsure reads go to a human.** A prompt can tell the model to return `"?"`
  (`config.UNSURE`) for a value it can't read with confidence. `""` still means
  a confident read of nothing there (e.g. no option circled). See the manual
  pass below.
- **Prefer one whole-page zone.** The model is capable enough to return every
  value from a single image, so a page costs one call. Crop into separate zones
  only when a read actually needs it.
- CSV columns are derived (base fields + each zone's columns + `points`); there
  is no column list to keep in sync.
- Adding an assignment = add a YAML file. The pipeline code never changes.
- **No file lists a class date.** The Canvas column defaults to the run's date,
  which the filename already carries, so one shared participation config serves
  every date. A per-date mapping would put dates back in a config and would
  need a line added per class; deriving it needs nothing. A quiz sidecar covers
  a single PDF, so it can name its assignment outright and carry an id.

**participation**: two zones, `andrew_id` (the ID boxes) and `response` (the
space below them) — two calls per page. The sample is split on purpose, to show
zones at work; one whole-page zone asking for both would do. CSV columns:
`response`, `points`.

The point hangs on **`andrew_id`**, not on `response`: participation is scored
on handing a sheet in, so what has to be read is the identity, which is a row
of printed boxes and reads reliably. `response` is transcribed for the record
and never scored. This is deliberate. Transcribing a hand-drawn table is not
reliable — on the 09-08 scan a handful of sheets came back with an empty
`response` despite being completely filled in, and the failures are all grids
and multi-column layouts in faint pencil, differing by model rather than being
fixed by one. Scoring on the transcript docked those students a point for the
reader's failure, which is the wrong way for this to break. An unsigned sheet
still scores nothing, because there is no identity to read.

## Dates

The class date comes from the **filename** (`YYYY-MM-DD`), never the LLM.
`runner._parse_date` throws if the name has no date, one in the wrong format, or
one that isn't a real calendar date. Name files like `2026-09-03.pdf`, and
quizzes like `2026-09-03-quiz.pdf` — rename a scan that arrives without a date
rather than loosening the regex, which exists to keep a 2-digit year out.

## Correctness hooks

- the config is schema-checked at load, which is the first thing a run does — so
  a bad region, a missing prompt, an unknown kind, a duplicate or reserved column
  name, or an unquoted `answer` fails before a page renders or a call is spent
- roster load asserts non-empty + unique ids
- ingest asserts pages rendered > 0; config rejects an `orient.tokens` list
  shorter than two or with one token inside another (the two-token minimum is
  only evidence if the tokens are independent)
- CSV `id_status` column flags rows to eyeball: `matched` | `loose` (second
  pass) | `manual` (a human said so) | `dropped` (a human said it is not a
  sheet) | `review` (nobody). A new value here must be added to
  `_validate_output`'s assertion, which pins the set, along with the companion
  assertion that a row without an id is `review` or `dropped`
- roster load asserts every Canvas identity column is present, and names the
  ones missing — an export taken without them is the likely cause
- the Canvas assignment name is checked at config load against Canvas's own
  rules (reserved column names, and the nine substrings like `Final Grade` it
  silently refuses), because that failure is invisible at upload time: the
  column is ignored and nobody is graded
- `_validate_output` asserts each row's keys equal the CSV header exactly
- CSV rows are one physical line each: whitespace in transcripts is collapsed on
  the way out (`_one_line`), so line-based tools and spreadsheet imports don't
  see a row break mid-answer. The verbatim read stays in `extractions.json`.
- badging (future) reconciles as a second source: add `badge_present`/`agrees`

## Orientation

The sheets are landscape, scanned into portrait pages, so every page needs a
90/270 rotation that can differ per page. `ingest._header_score` OCRs **only
the `orient` region** at each of the four rotations and counts printed header
tokens; the rotation that reads them is upright. Two tokens are required — one
match is as likely to be OCR noise ("aaammt" contains "mm") as a real label.
That minimum only means something if the tokens are independent evidence, so
**no token may contain another** — `date` alongside `dat` let a single word
score two and clear the bar alone. A test pins this.

**Which words to look for is config, not code** (`orient: {tokens: [...]}`,
defaulting to `config.HEADER_TOKENS`), because what a form prints is a property
of the form. The default list is the participation sheet's — `andrew`, `dat`,
`mm`, `dd` — and a form without a date field can never score two of them. The
09-15 quiz is exactly that: on the default tokens 1 of 12 pages read a header,
and that one only because OCR garbled something containing `dat`, which is the
noise the two-token rule exists to reject. Pointed at its own printed words
(`instructions`, `andrew`, `multiple`, `correct`, all off the instructions
line) it reads 12 of 12. `config._tokens` holds a task-supplied list to the
same independence rule as the default.

Cropping before OCR is the whole point, and it inverts an earlier measurement
in this repo: read page-wide, `--psm 11` scored 0 everywhere on faint scans, so
`--psm 6` was chosen. On a crop of two printed lines, psm 11 is the right mode
and a 3x upscale is affordable. On the 09-03 participation scan that took
orientation from 2/10 to 10/10 on the pages that had landed in review — they
were being handed to the vision model 180 degrees out by the OSD fallback,
which reads an ID box into confident nonsense.

A page that still reads nothing is settled in two more steps rather than handed
straight to OSD:

- **A second look.** `ingest._rotation` sweeps the four rotations once per mode
  in `HEADER_MODES` (`--psm 11`, then `--psm 6`), stopping at the first mode
  that clears the minimum. The modes segment the crop differently and so read
  different pages; the second sweep costs nothing on a page the first settled.
- **The document majority.** `ingest.majority_rotation` — the rotation most
  pages resolved to. A scan is one stack of sheets through one feeder, so the
  odd unreadable page is almost always turned like its neighbours, which beats
  guessing from its handwriting.

The majority is a fact about the whole document, so it can't exist inside a
per-page function — the same shape as `_rematch_unmatched`. `runner.run`
therefore ingests in two passes: `_get_rotation` per page (cached in the page's
`rotation.json`, `null` included, so a re-run doesn't re-sweep), then the
majority, then `_get_page` at the decided angle. **0 is a real rotation**, so
"undecided" is `None` throughout and never a falsy test — `rotations.get(i) or
fallback` would silently re-turn an already-upright page.

OSD is now the last resort only when *no* page in the document read a header —
i.e. the `orient` region is wrong for that form. It stays because it is the
only thing left to try, not because it is good: it reads handwriting and is
what put those 09-03 pages 180 degrees out.

## Second-pass identity matching

`runner._rematch_unmatched` runs after every page has been judged, because it
needs a fact that doesn't exist until then: **which students already have a
sheet**. A review row is then matched against only the ids nobody claimed —
usually a handful instead of 91. That shrunken candidate set, not a bigger
number, is what makes reaching further safe: the id of a student who already
has a sheet can no longer be stolen, which is exactly what raising
`MAX_DISTANCE` globally would invite.

Because a small candidate set also means noise has less competition, a winner
still has to look like an answer (`roster.rematch`):

- `LOOSE_DISTANCE` (4.0) — absolute ceiling
- `LOOSE_RATIO` (0.5) — and no further than half the longer string, so a short
  read can't travel; `wxyz` never becomes `rdas`
- `AMBIGUITY` (1.0) — must beat the runner-up by this, else it's a coin flip
- `OVERLAP_MARGIN` (2) — when two candidates are the *same* distance away, edit
  distance can't separate them, so the one sharing this many more characters
  (order-free, via `roster._overlap`) wins. `rmoan` is 3 from both `jmorgan`
  and `rdas`; it shares 5 characters with the first and 2 with the second.

Rows are assigned best-first and claim as they go, so a contested id goes to
the better fit rather than to whichever page was scanned first. Anything left
stays `review`: a blank read is never credited (that guard matters *more* here,
where one unmatched student makes an empty read trivially "closest"), and a
correct read of an id that simply isn't on the roster stays unidentified.

A resolved row is labelled `id_status=loose`, never promoted to `matched` — it
is a defensible guess, and the human skimming the CSV should know which rows
were guessed. Ranking also breaks distance ties by shared characters in pass 1,
which makes the winner deterministic instead of dependent on roster order.

## The manual pass

`review.resolve` runs after both automatic passes, for the same reason
`_rematch_unmatched` does: it needs what they leave behind — the rows still
unidentified, and the ids no page has claimed. It shows the page number, the
raw read, an excerpt of the transcript and the path to the saved page image, so
the sheet can be checked without hunting for it, then offers the closest
unclaimed ids by the same ranking pass 2 uses.

**Only unclaimed ids are on offer**, and each answer claims as it is given. So
the manual pass cannot do the one thing the automatic passes work hardest to
prevent — credit two sheets to one student. `review.duplicates` runs straight
afterwards anyway and logs any id on two rows. Nothing should ever reach it;
it exists because double-crediting is the single error a gradebook carries
silently, and `_validate_output`'s assertion is the hard stop behind it.

A row settled here is `manual`, never promoted to `matched`, for the same
reason `loose` is not: the CSV should say who decided. A skipped row stays
`review` and nobody is credited.

**Not every page is a sheet.** A cover page, a blank, or a scanning artefact is
answered with `d` and becomes `dropped`: it keeps its row, so the CSV still
accounts for every page of the PDF and nothing vanishes silently, but it scores
nothing and stops holding the upload back. Dropping is distinct from skipping
on purpose - skipping means "later" and keeps the upload waiting, dropping
means "never" and releases it. A drop claims no student, so the id stays
available to the page that really does belong to them.

**Unsure values are asked first** (`review.unsure`), before any judgment: every
value read as `"?"` is shown with its page image and the human types what the
sheet shows (`-` for no valid answer). That is transcription, so scoring then
treats it like any other read, and the answer key is never shown. It is not
saved - the cached reads stay the model's - so a re-run asks again, like the
identity pass. A scored field still `"?"` afterwards leaves the row's `points`
blank rather than scoring a guess, and holds back the Canvas file just as an
unidentified sheet does. An unscored field left `"?"` holds nothing.

**A run with no terminal skips the pass** rather than blocking on input nobody
can give — `sys.stdin.isatty()`. Ctrl-D and Ctrl-C end the pass and leave the
rest in review, so an interrupted run still writes everything already settled.

**The upload waits for the queue to empty.** While any row is `review`, the
results CSV is written and the Canvas file is not, and the run names the pages
holding it up. A student nobody named would otherwise go up as absent, which is
a wrong grade wearing the same face as a right one. Naming them and re-running
costs no model calls: only reads are cached, and no read has changed.

## Canvas upload

`canvas.write` writes `<stem>-canvas.csv` beside the results CSV, in the format
Canvas's gradebook importer reads. The results CSV is unchanged and stays the
spot-check artifact with the transcripts; this one is the thing to upload.

- **Identity is copied from the roster verbatim**, under Canvas's own column
  names. Canvas resolves a row by its numeric user id, so no upload can fall
  through to matching students by how their name is spelled. Storing those
  columns under Canvas's spelling is what makes the writer a straight copy
  rather than a mapping table that could drift.
- **One row per enrolled student, not one per page.** Canvas is being told the
  whole class's score, so a student with no sheet gets `absent` (default `0`;
  `""` leaves their existing grade alone) instead of being left out.
- **The column is named after the run's date**, which the filename already
  carries — see the config contract. A task needing another shape sets
  `canvas.assignment`, where `{date}` is substituted; an id in it (`Quiz 1
  (1029812)`) targets an assignment that already exists, because Canvas
  resolves the id before it looks at the title.
- **The `Points Possible` row is written by default**, at 1, because that is
  the shape Canvas's own export has and participation is worth a point. Set
  `points_possible: ""` to drop the row, which is what an assignment that
  already exists wants, since the row overwrites the value Canvas holds.
- A roster student with no Canvas identity (on the roster, in no export) still
  matches a sheet but is left out of the upload, with a warning. Dropping them
  from the roster instead would make their sheet unmatchable.

## Caching

Each PDF gets a run folder under `logs/`, named `<stem>-<pdf hash>-<reads hash>`.
The reads hash covers what affects a *read* (the zones' name, kind, region,
prompt and field names, plus the `orient` region and tokens, which decide how
the page is turned before anything is cropped) and deliberately not
`answer`/`points`: editing a region, a prompt or an orientation setting
re-reads,
while fixing an answer key re-grades cached transcripts in about a second with no
model calls. Keying the folder on zones also keeps the saved crops and the reads
in a folder consistent with each other. Old generations are never pruned; `logs/`
is safe to delete. Only the sample's run folder is committed: every other run
holds real students' pages, so `.gitignore` keeps it local.

## Run log

`run.log` opens with the run's parameters (config, date, roster size, pages,
workers, and the provider/model the reads will come from) and closes with a
`--- summary ---` section: wall time, model calls and tokens, then the row
counts. Each stage also logs how long it took, so a slow run says where the
time went. Tokens are counted in `vision._count`, which reads whichever field
names the provider's SDK uses; `0 calls` means every page came from the cache,
which is how a re-grade is told apart from a re-read.

## Vision providers

`pipeline/vision.py` is provider-agnostic: `anthropic` | `openai` | `gemini` |
`openrouter` | `lmstudio` | `bedrock`, picked via `config.VISION_PROVIDER` (or the
`VISION_PROVIDER` / `VISION_MODEL` env vars). One client per provider is created once and reused
(`vision._get_client`) — never construct a fresh SDK client per call, that
breaks concurrent extraction. `openrouter` reaches any vendor's model — including
free open-weights ones — through one OpenAI-compatible endpoint, so its ids are
`<vendor>/<model>`; `lmstudio` talks to a local LM Studio server instead of a
cloud API. `bedrock` uses Boto3 Converse with the standard AWS credential chain;
the default is `qwen.qwen3-vl-235b-a22b`. Configure the AWS region and optionally
override `VISION_MODEL` with a vision model or inference profile ID/ARN.
See `README.md` for setup/usage per provider.

Failures: transient API errors (429/5xx/overloaded/timeout) retry with
exponential backoff, and a reply that isn't usable JSON gets exactly one more
try — models occasionally ramble or truncate, and if a second ask doesn't fix
it, more calls won't. Anything else (e.g. a 401) fails fast. A page that still
fails is logged with its traceback and dropped from the CSV, and is never
cached, so re-running the same command retries only that page.

"""Stage 1 (ingest): PDF -> normalized page images.

Ingest is the one stage with no vision model in it, so it can be tested for
real - no API key, no network, deterministic. The fixture is a synthetic scan
(printed sheets, made-up Andrew IDs) whose pages sit landscape in the PDF,
turned and tilted, and have to come out upright, which is precisely the job
ingest exists to do.

    python -m unittest discover -s tests -t . -v

Ingesting a page sweeps OCR over four rotations - twice, if the first mode
can't read it - plus a deskew search (~5s), so the pages are ingested once for
the whole class.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fitz
import numpy as np
import pytesseract
from PIL import Image

from pipeline import config, ingest, runner

FIXTURE = Path(__file__).parent / "fixtures" / "quiz-scan-3-pages.pdf"
PAGE_COUNT = 3
PRINTED_PHRASE = "circle the letter"  # printed on every page of the fixture


class TestIngest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = ingest.load_pages(FIXTURE)  # also covers load_pages itself

    # --- the contract: pages come out upright -------------------------------

    def test_source_pages_are_landscape(self):
        """Guards the test below: if the fixture were already portrait, the
        'output is portrait' assertion would pass without ingest doing anything."""
        with fitz.open(FIXTURE) as doc:
            for i in range(PAGE_COUNT):
                raw = ingest._render(doc[i])
                self.assertGreater(raw.width, raw.height, f"page {i + 1} is not landscape")

    def test_pages_come_out_portrait(self):
        for page in self.pages:
            self.assertGreater(page.image.height, page.image.width,
                               f"page {page.index} was not rotated upright")

    def test_pages_are_upright_not_upside_down(self):
        """Aspect ratio can't tell 0 from 180 degrees, so read the page: printed
        text only OCRs when it is the right way up. Checked against the text
        rather than ingest's own header heuristic, so a bug in that heuristic
        can't make this pass."""
        for page in self.pages:
            text = pytesseract.image_to_string(page.image).lower()
            self.assertIn(PRINTED_PHRASE, text,
                          f"page {page.index} does not read as upright")

    def test_upside_down_would_be_caught(self):
        """Guards the test above: the phrase must actually discriminate, or that
        assertion would pass on any orientation."""
        upside_down = self.pages[0].image.rotate(180, expand=True)
        self.assertNotIn(PRINTED_PHRASE, pytesseract.image_to_string(upside_down).lower())

    def test_pages_are_deskewed(self):
        """After deskewing, the residual tilt must be below the threshold that
        would have triggered another correction."""
        for page in self.pages:
            residual = ingest._skew_angle(np.array(page.image.convert("L")))
            self.assertLess(abs(residual), ingest.DESKEW_MIN,
                            f"page {page.index} still tilted by {residual:.2f} deg")

    # --- shape of the output ------------------------------------------------

    def test_page_count(self):
        self.assertEqual(ingest.page_count(FIXTURE), PAGE_COUNT)

    def test_every_page_is_returned_in_order(self):
        self.assertEqual([p.index for p in self.pages], list(range(1, PAGE_COUNT + 1)))

    def test_images_are_rgb_and_full_size(self):
        for page in self.pages:
            self.assertEqual(page.image.mode, "RGB")
            # 200 DPI on letter paper; well under this means a broken render
            self.assertGreater(min(page.image.size), 1000, f"page {page.index} too small")

    # --- properties the runner's cache depends on ---------------------------

    def test_load_page_is_deterministic(self):
        """The cache stores a page image and reuses it across runs, which is
        only sound if ingesting the same page twice gives the same pixels."""
        again = ingest.load_page(FIXTURE, 1)
        self.assertEqual(again.image.tobytes(), self.pages[0].image.tobytes())

    def test_load_page_is_one_based(self):
        self.assertEqual(ingest.load_page(FIXTURE, 1).image.tobytes(),
                         self.pages[0].image.tobytes())
        with self.assertRaises(IndexError):
            ingest.load_page(FIXTURE, PAGE_COUNT + 1)

    # --- failure modes ------------------------------------------------------

    def test_missing_pdf_is_rejected(self):
        missing = FIXTURE.parent / "does-not-exist.pdf"
        with self.assertRaises(AssertionError):
            ingest.page_count(missing)
        with self.assertRaises(AssertionError):
            ingest.load_page(missing, 1)


class TestHeaderZone(unittest.TestCase):
    """Orientation reads the header zone, whose coordinates live in config."""

    def test_the_zone_is_what_gets_read(self):
        """Score the fixture's upright page against the header zone and against
        a strip of empty page. Only the first can see the labels."""
        page = ingest.load_page(FIXTURE, 1).image
        self.assertGreaterEqual(
            ingest._header_score(page, config.ORIENT_REGION), ingest.HEADER_MIN_SCORE)
        self.assertLess(
            ingest._header_score(page, (0.0, 0.8, 1.0, 0.2)), ingest.HEADER_MIN_SCORE)

    def test_one_token_is_not_enough(self):
        """A lone match is as likely OCR noise as a label, so it must not
        decide a rotation on its own."""
        self.assertGreater(ingest.HEADER_MIN_SCORE, 1)

    def test_no_header_token_contains_another(self):
        """The minimum counts distinct tokens as independent evidence, so a
        token containing another would let one OCR'd word score twice and clear
        the bar alone - which is the thing the minimum exists to stop."""
        for a in config.HEADER_TOKENS:
            for b in config.HEADER_TOKENS:
                if a != b:
                    self.assertNotIn(b, a, f"{b!r} is a substring of {a!r}")

    def test_a_task_can_override_the_zone(self):
        cfg = self._write("orient:\n  region: [0.1, 0.0, 0.8, 0.25]\n")
        self.assertEqual(cfg.orient, (0.1, 0.0, 0.8, 0.25))

    def test_the_default_applies_when_a_task_says_nothing(self):
        self.assertEqual(self._write("").orient, config.ORIENT_REGION)
        self.assertEqual(self._write("").tokens, config.HEADER_TOKENS)

    def test_a_task_can_override_the_tokens(self):
        """A form that prints different labels is a config change: the quiz has
        no date field, so the participation sheet's tokens can never score."""
        cfg = self._write('orient:\n  tokens: ["instructions", "andrew"]\n')
        self.assertEqual(cfg.tokens, ("instructions", "andrew"))

    def test_task_tokens_must_be_independent_evidence(self):
        """Same rule the default list is held to, enforced at config load."""
        with self.assertRaises(AssertionError):
            self._write('orient:\n  tokens: ["date", "dat"]\n')

    def test_one_task_token_cannot_clear_the_bar_alone(self):
        with self.assertRaises(AssertionError):
            self._write('orient:\n  tokens: ["andrew"]\n')

    def test_a_zone_off_the_page_is_rejected(self):
        with self.assertRaises(AssertionError):
            self._write("orient:\n  region: [0.5, 0.0, 0.8, 0.2]\n")

    def _write(self, orient_yaml):
        """A minimal task config, plus whatever orient block is under test."""
        d = Path(tempfile.mkdtemp())
        (d / "t.yaml").write_text(
            "zones:\n"
            "  - name: andrew_id\n"
            "    kind: boxed_id\n"
            "    region: [0.0, 0.0, 1.0, 0.1]\n"
            "    prompt: read it\n" + orient_yaml)
        return config.load(d / "t.pdf")


class TestPagesThatWillNotRead(unittest.TestCase):
    """What happens to a page whose header OCR can't see: a second look in a
    different segmentation mode, then the rotation the rest of the stack got."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.mixed = Path(cls.tmp.name) / "one-header-one-blank.pdf"
        doc = fitz.open()
        doc.new_page().insert_text((72, 120), "AndrewID:    Date (MM/DD):", fontsize=20)
        doc.new_page()  # no header at all
        doc.save(cls.mixed)
        doc.close()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # --- the escalation ladder ----------------------------------------------

    def test_a_second_look_settles_a_page_the_first_mode_misses(self):
        """Mode 1 reading nothing must not end the page: mode 2 segments the
        crop differently and gets its own sweep of the four rotations."""
        seen = []

        def scored(img, region, mode=ingest.HEADER_MODES[0], tokens=()):
            seen.append(mode)
            return ingest.HEADER_MIN_SCORE if len(seen) == 6 else 0  # mode 2, 90deg

        with mock.patch.object(ingest, "_header_score", scored):
            rotation = ingest._rotation(Image.new("RGB", (80, 100)), config.ORIENT_REGION)
        self.assertEqual(rotation, 90)
        self.assertEqual(seen, [ingest.HEADER_MODES[0]] * 4 + [ingest.HEADER_MODES[1]] * 4)

    def test_the_second_look_is_skipped_when_the_first_settles_it(self):
        """The extra sweep costs nothing on the pages that already read."""
        seen = []

        def scored(img, region, mode=ingest.HEADER_MODES[0], tokens=()):
            seen.append(mode)
            return ingest.HEADER_MIN_SCORE if len(seen) == 1 else 0  # mode 1, 0deg

        with mock.patch.object(ingest, "_header_score", scored):
            self.assertEqual(
                ingest._rotation(Image.new("RGB", (80, 100)), config.ORIENT_REGION), 0)
        self.assertEqual(seen, [ingest.HEADER_MODES[0]] * 4)

    def test_a_page_no_mode_can_read_is_unresolved(self):
        with mock.patch.object(ingest, "_header_score", lambda *a, **k: 0):
            self.assertIsNone(
                ingest._rotation(Image.new("RGB", (80, 100)), config.ORIENT_REGION))

    # --- the document majority ----------------------------------------------

    def test_the_majority_is_how_most_pages_were_turned(self):
        self.assertEqual(ingest.majority_rotation([90, None, 90, 270]), 90)

    def test_pages_that_did_not_read_do_not_vote(self):
        self.assertEqual(ingest.majority_rotation([None, None, 270]), 270)

    def test_there_is_no_majority_when_nothing_read(self):
        self.assertIsNone(ingest.majority_rotation([None, None]))
        self.assertIsNone(ingest.majority_rotation([]))

    def test_a_decided_rotation_is_applied_without_reading_the_page(self):
        """A page settled by the majority must not pay for another OCR sweep."""
        img = Image.new("RGB", (80, 100))
        with mock.patch.object(ingest, "_rotation",
                               side_effect=AssertionError("re-read the page")):
            self.assertEqual(ingest._orient(img, config.ORIENT_REGION, 90).size, (100, 80))

    def test_zero_is_a_decision_not_an_absent_one(self):
        """0 is a real rotation. Treated as falsy anywhere on the path it would
        silently become 'undecided' and re-read, or take the majority instead."""
        img = Image.new("RGB", (80, 100))
        with mock.patch.object(ingest, "_rotation",
                               side_effect=AssertionError("re-read the page")):
            self.assertEqual(ingest._orient(img, config.ORIENT_REGION, 0).size, (80, 100))

    def test_an_unreadable_page_takes_the_majority_instead_of_osd(self):
        """End to end: page 2 has no header, so it must be turned like page 1.
        OSD is patched to fail loudly, since reaching it is the old behaviour."""
        self.assertIsNotNone(ingest.page_rotation(self.mixed, 1), "page 1 must read")
        self.assertIsNone(ingest.page_rotation(self.mixed, 2), "page 2 must not read")
        with mock.patch.object(ingest, "_osd_orient",
                               side_effect=AssertionError("fell through to OSD")):
            pages = ingest.load_pages(self.mixed)
        self.assertEqual([p.index for p in pages], [1, 2])

    # --- the property the runner's cache depends on --------------------------

    def test_a_decided_rotation_is_cached_and_reused(self):
        cache = Path(tempfile.mkdtemp())
        first = runner._get_rotation(cache, self.mixed, 1, config.ORIENT_REGION,
                                     config.HEADER_TOKENS)
        with mock.patch.object(ingest, "page_rotation",
                               side_effect=AssertionError("re-swept a cached page")):
            self.assertEqual(runner._get_rotation(cache, self.mixed, 1,
                                                  config.ORIENT_REGION,
                                                  config.HEADER_TOKENS), first)

    def test_an_unreadable_page_is_cached_as_unreadable(self):
        """None has to survive the cache, or every re-run re-sweeps the pages
        that were never going to read."""
        cache = Path(tempfile.mkdtemp())
        self.assertIsNone(runner._get_rotation(cache, self.mixed, 2,
                                              config.ORIENT_REGION, config.HEADER_TOKENS))
        with mock.patch.object(ingest, "page_rotation",
                               side_effect=AssertionError("re-swept a cached page")):
            self.assertIsNone(
                runner._get_rotation(cache, self.mixed, 2, config.ORIENT_REGION,
                                     config.HEADER_TOKENS))


if __name__ == "__main__":
    unittest.main()

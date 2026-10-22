"""Stage 2 (extract): crop a zone, read it, clean the value.

No vision model is needed - the JSON reply is supplied directly, which puts the
cropping and cleaning under test rather than the model. Reading stays separate
from judgment, so nothing here scores anything.

Every Andrew ID and name in this file is made up; none is a real student.

    python -m unittest discover -s tests -t . -v
"""

import unittest
from unittest import mock

from PIL import Image

from pipeline import config, extract
from pipeline.ingest import Page

PAGE = Page(index=1, image=Image.new("RGB", (400, 200), "white"))
SHEET = [config.Field("andrew_id"), config.Field("response", points=1)]


def zone(**kw) -> config.Zone:
    d = dict(name="sheet", kind="freetext", region=(0.0, 0.0, 1.0, 1.0),
             prompt="read the sheet")
    d.update(kw)
    return config.Zone(**d)


def read(z: config.Zone, reply: dict, page: Page = PAGE) -> dict:
    """extract_zone with the model's JSON reply supplied; returns {column: value}."""
    with mock.patch("pipeline.vision.read_json", return_value=reply):
        return {c: ex.value for c, ex in extract.extract_zone(page, z).items()}


class TestCrop(unittest.TestCase):
    """A zone's region is the image the model is asked about."""

    def test_a_fractional_region_becomes_the_pixel_box_it_names(self):
        """Position, not just size: a crop that landed on the wrong half would
        confidently read the wrong answer."""
        img = Image.new("RGB", (400, 200), "white")
        img.paste(Image.new("RGB", (200, 200), "black"), (200, 0))
        page = Page(index=1, image=img)
        left = extract.crop_zone(page, zone(region=(0.0, 0.0, 0.5, 1.0)))
        right = extract.crop_zone(page, zone(region=(0.5, 0.0, 0.5, 1.0)))
        self.assertEqual(left.size, (200, 200))
        self.assertEqual(left.convert("L").getextrema(), (255, 255), "took the wrong half")
        self.assertEqual(right.convert("L").getextrema(), (0, 0), "took the wrong half")

    def test_a_whole_page_zone_is_the_whole_page(self):
        self.assertEqual(extract.crop_zone(PAGE, zone()).size, PAGE.image.size)


class TestColumns(unittest.TestCase):
    """One crop, one call, one column per value the prompt asks for."""

    def test_each_field_the_prompt_asks_for_becomes_a_column(self):
        got = read(zone(fields=SHEET), {"andrew_id": "jdoe", "response": "an answer"})
        self.assertEqual(got, {"andrew_id": "jdoe", "response": "an answer"})

    def test_a_single_value_zone_is_named_after_the_zone(self):
        got = read(zone(name="q1", kind="bubble", region=(0.0, 0.2, 1.0, 0.1)),
                   {"value": "c"})
        self.assertEqual(got, {"q1": "c"})

    def test_a_field_the_model_omitted_comes_back_empty(self):
        """Negative: a missing key must not raise - the row still has to reach
        the CSV to be corrected, not vanish with an exception."""
        got = read(zone(fields=SHEET), {"andrew_id": "jdoe"})
        self.assertEqual(got["response"], "")


class TestCleaning(unittest.TestCase):
    """Each kind says how its read is normalized."""

    def test_a_boxed_id_loses_its_spaces_and_case(self):
        got = read(zone(name="andrew_id", kind="boxed_id", region=(0.0, 0.0, 0.5, 0.1)),
                   {"value": " J D o e "})
        self.assertEqual(got["andrew_id"], "jdoe")

    def test_an_id_read_off_a_freetext_page_is_still_normalized(self):
        """A whole-page zone declares one kind, but the id still has no spaces."""
        got = read(zone(fields=SHEET), {"andrew_id": "J Doe", "response": "x"})
        self.assertEqual(got["andrew_id"], "jdoe")

    def test_a_bubble_is_lowercased(self):
        got = read(zone(name="q1", kind="bubble", region=(0.0, 0.2, 1.0, 0.1)),
                   {"value": "C"})
        self.assertEqual(got["q1"], "c")

    def test_a_transcript_is_left_as_written(self):
        """Negative: freetext is what a human will grade, so cleaning must not
        lowercase it or collapse the writing inside it."""
        got = read(zone(fields=SHEET),
                   {"andrew_id": "jdoe", "response": "  Two  Words Here "})
        self.assertEqual(got["response"], "Two  Words Here")


if __name__ == "__main__":
    unittest.main()

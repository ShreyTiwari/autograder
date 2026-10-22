"""The token counts in the run summary.

Every SDK reports usage under its own field names, and a name that stopped
matching would quietly report zero rather than fail. This pins the shapes.

    python -m unittest discover -s tests -t . -v
"""

import unittest
from types import SimpleNamespace as Usage

from pipeline import vision


class TestUsage(unittest.TestCase):

    def setUp(self):
        vision._usage.update(calls=0, input=0, output=0)

    def test_each_provider_shape_is_counted(self):
        vision._count(Usage(input_tokens=10, output_tokens=1))  # anthropic
        vision._count(Usage(prompt_tokens=20, completion_tokens=2))  # openai family
        vision._count(Usage(prompt_token_count=30, candidates_token_count=3))  # gemini
        vision._count({"inputTokens": 40, "outputTokens": 4})  # bedrock
        self.assertEqual(vision.usage(), {"calls": 4, "input": 100, "output": 10})

    def test_a_provider_reporting_no_usage_still_counts_the_call(self):
        vision._count(None)
        self.assertEqual(vision.usage(), {"calls": 1, "input": 0, "output": 0})

"""Bedrock requests and error handling, with no network or AWS credentials."""

import io
import os
import unittest
from contextlib import ExitStack
from unittest import mock

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.stub import ANY, Stubber
from PIL import Image

from pipeline import config, vision


MODEL = "test-vision-model"
RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [
        {"text": '{"andrew_id":'},
        {"toolUse": {"toolUseId": "unused", "name": "unused", "input": {}}},
        {"text": '"example"}'},
    ]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 12, "outputTokens": 5, "totalTokens": 17},
    "metrics": {"latencyMs": 10},
}


class TestBedrock(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict(os.environ, {
            "VISION_PROVIDER": "bedrock", "VISION_MODEL": MODEL,
            "AWS_REGION": "us-east-1", "AWS_DEFAULT_REGION": "us-west-2",
        }, clear=True))
        self.stack.enter_context(mock.patch.object(vision, "_clients", {}))
        self.stack.enter_context(mock.patch.object(vision, "_usage", {"calls": 0, "input": 0, "output": 0}))
        self.sleep = self.stack.enter_context(mock.patch("pipeline.vision.time.sleep"))
        self.client = boto3.client("bedrock-runtime", region_name="us-east-1",
                                  aws_access_key_id="testing", aws_secret_access_key="testing")
        self.addCleanup(self.client.close)
        self.session = self.stack.enter_context(mock.patch("boto3.Session"))
        self.session.return_value.client.return_value = self.client
        self.stubber = self.stack.enter_context(Stubber(self.client))
        self.image = Image.new("RGBA", (24, 16), "white")
        self.params = {
            "modelId": MODEL,
            "messages": [{"role": "user", "content": [
                {"image": {"format": "png", "source": {"bytes": ANY}}},
                {"text": "read the id"},
            ]}],
            "inferenceConfig": {"maxTokens": 128},
        }

    def read(self):
        return vision.read_json(self.image, "read the id", max_tokens=128)

    def test_converse_request_image_response_usage_and_client_reuse(self):
        png = vision._png_bytes(self.image)
        with Image.open(io.BytesIO(png)) as image:
            self.assertEqual((image.format, image.mode, image.size), ("PNG", "RGB", (24, 16)))
        self.params["messages"][0]["content"][0]["image"]["source"]["bytes"] = png
        for _ in range(2):
            self.stubber.add_response("converse", RESPONSE, self.params)
            self.assertEqual(self.read(), {"andrew_id": "example"})
        self.assertEqual(vision.usage(), {"calls": 2, "input": 24, "output": 10})
        self.session.assert_called_once_with()
        args, kwargs = self.session.return_value.client.call_args
        self.assertEqual(args, ("bedrock-runtime",))
        self.assertEqual(kwargs["region_name"], "us-east-1")
        self.assertEqual(kwargs["config"].retries, {"total_max_attempts": 1})
        self.assertEqual(kwargs["config"].read_timeout, 600)
        self.sleep.assert_not_called()
        self.stubber.assert_no_pending_responses()

    def test_region_falls_back_to_default_region_then_profile(self):
        del os.environ["AWS_REGION"]
        for expected_region in ("us-west-2", None):
            vision._clients.clear()
            self.stubber.add_response("converse", RESPONSE, self.params)
            self.read()
            self.assertEqual(self.session.return_value.client.call_args.kwargs["region_name"],
                             expected_region)
            os.environ.pop("AWS_DEFAULT_REGION", None)

    def test_transient_aws_errors_retry_and_only_count_success(self):
        for code, status in (("ThrottlingException", 429),
                             ("InternalServerException", 500),
                             ("ServiceUnavailableException", 503),
                             ("ModelTimeoutException", 408),
                             ("ModelNotReadyException", 429)):
            with self.subTest(code=code):
                self.sleep.reset_mock()
                self.stubber.add_client_error("converse", code, "try again", status,
                                              expected_params=self.params)
                self.stubber.add_response("converse", RESPONSE, self.params)
                self.assertEqual(self.read(), {"andrew_id": "example"})
                self.sleep.assert_called_once_with(vision.BACKOFF)
        self.assertEqual(vision.usage()["calls"], 5)

    def test_permanent_aws_errors_fail_without_retry(self):
        for code in ("AccessDeniedException", "ValidationException", "ResourceNotFoundException"):
            with self.subTest(code=code):
                self.stubber.add_client_error("converse", code, "invalid request", 400,
                                              expected_params=self.params)
                with self.assertRaises(ClientError):
                    self.read()
        self.sleep.assert_not_called()
        self.assertEqual(vision.usage()["calls"], 0)

    def test_exhausted_retries_propagate_the_error(self):
        for _ in range(vision.RETRIES):
            self.stubber.add_client_error("converse", "ThrottlingException", "try again", 429,
                                          expected_params=self.params)
        with self.assertRaises(ClientError):
            self.read()
        self.assertEqual(self.sleep.call_args_list,
                         [mock.call(vision.BACKOFF * 2 ** i) for i in range(vision.RETRIES - 1)])
        self.stubber.assert_no_pending_responses()
        self.assertEqual(vision.usage()["calls"], 0)

    def test_missing_credentials_fail_without_retry(self):
        self.session.return_value.client.side_effect = NoCredentialsError()
        with self.assertRaises(NoCredentialsError):
            self.read()
        self.sleep.assert_not_called()

    def test_missing_model_explains_configuration_before_creating_client(self):
        del os.environ["VISION_MODEL"]
        with mock.patch.dict(config.VISION_MODEL, {"bedrock": None}), \
                self.assertRaisesRegex(ValueError, "set VISION_MODEL"):
            self.read()
        self.session.assert_not_called()

    def test_bedrock_defaults_to_qwen_when_model_is_not_overridden(self):
        del os.environ["VISION_MODEL"]
        self.params["modelId"] = "qwen.qwen3-vl-235b-a22b"
        self.stubber.add_response("converse", RESPONSE, self.params)
        self.assertEqual(self.read(), {"andrew_id": "example"})
        self.stubber.assert_no_pending_responses()


if __name__ == "__main__":
    unittest.main()

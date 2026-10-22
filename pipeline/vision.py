"""Vision-model client - the one place that talks to an LLM.

Provider-agnostic: pick anthropic | openai | gemini | openrouter | lmstudio | bedrock
via config (or the VISION_PROVIDER / VISION_MODEL env vars). Each SDK is
imported lazily, so only the chosen provider needs to be installed and hold an
API key. openrouter reaches every vendor's models (and free open-weights ones)
through one OpenAI-compatible endpoint; lmstudio talks to a local LM Studio
server instead - no key needed, just the server running. Bedrock uses the
standard AWS credential chain and its Converse API.

Kept thin and transcription-only: it turns an image + prompt into text. Parsing
and any judgment stay in the caller (extract / score).
"""

import base64
import io
import json
import logging
import os
import threading
import time

from . import config

log = logging.getLogger("pipeline")  # the runner routes this to run.log

# One client per provider, created once and reused. httpx-based SDK clients are
# thread-safe for concurrent requests; the danger is a transient client getting
# garbage-collected mid-run and closing a shared transport, so we hold one.
_clients: dict = {}
_clients_lock = threading.Lock()

# Token usage for the run's summary line. Every provider reports it, under a
# different name; one that reports none still has its call counted.
_usage = {"calls": 0, "input": 0, "output": 0}
_usage_lock = threading.Lock()

RETRIES = 4  # transient API errors; these produced no model output
JSON_RETRIES = 2  # a malformed reply gets one more try, then the page fails
BACKOFF = 2.0  # seconds; doubled each attempt
# Substrings that mark a retryable (transient) API error across providers.
_TRANSIENT = ("503", "429", "500", "502", "504", "overloaded",
              "unavailable", "high demand", "timeout", "rate limit",
              "throttlingexception", "internalserverexception", "modelnotreadyexception")


def _is_transient(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(t in s for t in _TRANSIENT)


def _get_client(provider: str, factory):
    """One client per provider, created once and reused (never GC'd mid-run,
    which can close a transport a request still needs - or, for the anonymous
    `Client().foo(...)` pattern, get collected before the call even completes)."""
    client = _clients.get(provider)
    if client is None:
        with _clients_lock:
            client = _clients.get(provider)
            if client is None:
                client = _clients[provider] = factory()
    return client


def usage() -> dict:
    """Model calls and tokens spent so far this run."""
    with _usage_lock:
        return dict(_usage)


def _count(u) -> None:
    """Add one call's usage to the run total, whatever the SDK names the fields."""
    def field(*names):
        get = u.get if isinstance(u, dict) else lambda n: getattr(u, n, None)
        return next((v for n in names if (v := get(n)) is not None), 0)
    with _usage_lock:
        _usage["calls"] += 1
        _usage["input"] += field("input_tokens", "prompt_tokens", "prompt_token_count",
                                 "inputTokens")
        _usage["output"] += field("output_tokens", "completion_tokens",
                                  "candidates_token_count", "outputTokens")


def target() -> tuple:
    """The provider and model a call would use now; env overrides config."""
    provider = os.environ.get("VISION_PROVIDER", config.VISION_PROVIDER).lower()
    return provider, (os.environ.get("VISION_MODEL")
                      or config.VISION_MODEL.get(provider))


def read_json(image, prompt: str, max_tokens: int = 8192) -> dict:
    """Ask the model for JSON and parse it, tolerating code fences / prose.

    A reply that isn't usable JSON gets exactly one more try - models
    occasionally ramble or truncate, and asking again usually fixes it; if it
    doesn't, more calls won't. Each bad reply is logged, so a zone that never
    parses can be diagnosed from run.log instead of just showing up as a
    failed page.
    """
    for attempt in range(JSON_RETRIES):
        text = read_image(image, prompt, max_tokens)
        try:
            return _parse_json(text)
        except ValueError as e:  # json.JSONDecodeError is a ValueError
            log.warning("unparseable reply (attempt %d/%d): %s",
                        attempt + 1, JSON_RETRIES, e)
            if attempt == JSON_RETRIES - 1:
                raise


def _parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in model reply: {text!r}")
    return json.loads(text[start : end + 1])


def read_image(image, prompt: str, max_tokens: int = 8192) -> str:
    """Send one image + prompt to the configured provider; return its text.

    Retries transient errors (overload / rate limit) with exponential backoff;
    the runner's cache-resume covers anything that still fails.
    """
    provider, model = target()
    if not model:
        raise ValueError(f"no model configured for provider {provider!r}; set VISION_MODEL")
    fn = {"anthropic": _anthropic, "openai": _openai, "gemini": _gemini,
          "openrouter": _openrouter, "lmstudio": _lmstudio, "bedrock": _bedrock}.get(provider)
    if fn is None:
        raise ValueError(f"unknown VISION_PROVIDER {provider!r}")
    for attempt in range(RETRIES):
        try:
            return fn(model, image, prompt, max_tokens)
        except Exception as e:
            if attempt == RETRIES - 1 or not _is_transient(e):
                raise
            time.sleep(BACKOFF * 2 ** attempt)


def _png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _anthropic(model, image, prompt, max_tokens) -> str:
    from anthropic import Anthropic
    client = _get_client("anthropic", Anthropic)
    b64 = base64.b64encode(_png_bytes(image)).decode()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    _count(msg.usage)
    return "".join(b.text for b in msg.content if b.type == "text")


def _bedrock(model, image, prompt, max_tokens) -> str:
    import boto3
    from botocore.config import Config

    # Session picks up AWS_PROFILE, environment credentials, SSO, or an IAM role.
    # read_image owns retries; keep SDK retries from multiplying requests.
    client = _get_client("bedrock", lambda: boto3.Session().client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
        config=Config(read_timeout=600, retries={"total_max_attempts": 1}),
    ))
    resp = client.converse(
        modelId=model,
        messages=[{"role": "user", "content": [
            {"image": {"format": "png", "source": {"bytes": _png_bytes(image)}}},
            {"text": prompt},
        ]}],
        inferenceConfig={"maxTokens": max_tokens},
    )
    _count(resp.get("usage"))
    return "".join(b["text"] for b in resp["output"]["message"]["content"] if "text" in b)


def _openai(model, image, prompt, max_tokens) -> str:
    from openai import OpenAI
    client = _get_client("openai", OpenAI)
    b64 = base64.b64encode(_png_bytes(image)).decode()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    )
    _count(resp.usage)
    return resp.choices[0].message.content


def _gemini(model, image, prompt, max_tokens) -> str:
    from google import genai
    from google.genai import types
    client = _get_client("gemini", genai.Client)
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=_png_bytes(image), mime_type="image/png"), prompt],
        config=types.GenerateContentConfig(max_output_tokens=max_tokens),
    )
    _count(resp.usage_metadata)
    return resp.text


def _openrouter(model, image, prompt, max_tokens) -> str:
    """Same OpenAI-compatible shape as _openai, pointed at OpenRouter, which
    fronts every vendor - so the model id is "<vendor>/<model>"."""
    from openai import OpenAI
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set")
    # max_retries=0: read_image owns the retry policy, so the SDK must not
    # silently multiply calls on top of it.
    client = _get_client("openrouter", lambda: OpenAI(
        base_url=config.OPENROUTER_BASE_URL, api_key=key, max_retries=0))
    b64 = base64.b64encode(_png_bytes(image)).decode()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    )
    if resp.choices is None:  # OpenRouter reports upstream failures in-band
        raise RuntimeError(f"openrouter returned no choices: {resp}")
    _count(resp.usage)
    return resp.choices[0].message.content


def _lmstudio(model, image, prompt, max_tokens) -> str:
    """Same OpenAI-compatible chat-completions shape as _openai, pointed at a
    local LM Studio server instead. No real API key needed."""
    from openai import OpenAI
    # max_retries=0: read_image owns the retry policy, so the SDK must not
    # silently multiply calls on top of it.
    client = _get_client("lmstudio", lambda: OpenAI(
        base_url=config.LMSTUDIO_BASE_URL, api_key="lm-studio",
        timeout=config.LMSTUDIO_TIMEOUT, max_retries=0))
    b64 = base64.b64encode(_png_bytes(image)).decode()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    )
    _count(resp.usage)
    return resp.choices[0].message.content

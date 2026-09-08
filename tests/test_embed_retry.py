"""Tests for GeminiEmbeddingProvider retry logic — no real API calls, no real sleeping.

These tests mock out `genai.Client.models.embed_content` /
`genai.Client.aio.models.embed_content` so no network request (and no API
quota) is ever used, and patch `time.sleep` / `asyncio.sleep` so the tests
run instantly instead of waiting through the real backoff delays.
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from google.genai import errors

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ipeecs_bot.llm_api.embed_method.embed_gemini import GeminiEmbeddingProvider


def make_provider() -> GeminiEmbeddingProvider:
    """Builds a provider with a dummy key. Client construction is offline/lazy,
    so this never touches the network."""
    return GeminiEmbeddingProvider(api_key="fake-key-for-test")


def rate_limit_error() -> errors.ClientError:
    return errors.ClientError(code=429, response_json={"message": "rate limited", "status": "RESOURCE_EXHAUSTED"})


def test_sync_retry_recovers_from_transient_error():
    """A transient (non-429) error should be retried and eventually succeed."""
    provider = make_provider()
    call = MagicMock(side_effect=[RuntimeError("temporary glitch"), "ok"])

    with patch("time.sleep") as mock_sleep:
        result = provider._call_with_retry(call)

    assert result == "ok"
    assert call.call_count == 2
    mock_sleep.assert_called_once()


def test_sync_retry_exhausts_and_raises():
    """After DEFAULT_MAX_RETRIES attempts, the original exception must propagate."""
    provider = make_provider()
    call = MagicMock(side_effect=RuntimeError("always fails"))

    with patch("time.sleep"):
        try:
            provider._call_with_retry(call)
            assert False, "expected RuntimeError to propagate"
        except RuntimeError as e:
            assert str(e) == "always fails"

    assert call.call_count == provider.DEFAULT_MAX_RETRIES


def test_sync_retry_waits_at_least_50s_on_rate_limit():
    """429 backoff must respect RATE_LIMIT_MIN_WAIT even if delay is smaller."""
    provider = make_provider()
    call = MagicMock(side_effect=[rate_limit_error(), "ok"])

    with patch("time.sleep") as mock_sleep:
        result = provider._call_with_retry(call, initial_delay=1.0)

    assert result == "ok"
    mock_sleep.assert_called_once_with(provider.RATE_LIMIT_MIN_WAIT)


def test_async_retry_recovers_from_transient_error():
    provider = make_provider()
    call = AsyncMock(side_effect=[RuntimeError("temporary glitch"), "ok"])

    async def run():
        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await provider._call_with_retry_async(call)
            return result, mock_sleep

    result, mock_sleep = asyncio.run(run())
    assert result == "ok"
    assert call.call_count == 2
    mock_sleep.assert_called_once()


def test_async_retry_exhausts_and_raises():
    provider = make_provider()
    call = AsyncMock(side_effect=RuntimeError("always fails"))

    async def run():
        with patch("asyncio.sleep", new=AsyncMock()):
            await provider._call_with_retry_async(call)

    try:
        asyncio.run(run())
        assert False, "expected RuntimeError to propagate"
    except RuntimeError as e:
        assert str(e) == "always fails"

    assert call.call_count == provider.DEFAULT_MAX_RETRIES


def test_embed_query_sync_uses_retry_helper_without_hitting_network():
    """Ensures embed_query_sync goes through the client without a real API call."""
    provider = make_provider()
    fake_embedding = MagicMock(values=[0.1, 0.2, 0.3])
    fake_response = MagicMock(embeddings=[fake_embedding])

    with patch.object(provider.client.models, "embed_content", return_value=fake_response) as mock_call:
        result = provider.embed_query_sync("測試問題")

    assert result == [0.1, 0.2, 0.3]
    mock_call.assert_called_once_with(model=provider.model_name, contents="測試問題")


if __name__ == "__main__":
    tests = [
        test_sync_retry_recovers_from_transient_error,
        test_sync_retry_exhausts_and_raises,
        test_sync_retry_waits_at_least_50s_on_rate_limit,
        test_async_retry_recovers_from_transient_error,
        test_async_retry_exhausts_and_raises,
        test_embed_query_sync_uses_retry_helper_without_hitting_network,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("All retry tests passed (no API quota used).")

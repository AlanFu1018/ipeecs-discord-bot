"""Tests for GeminiLLMProvider 503 retry logic — no real API calls, no real sleeping.

These tests mock out `genai.Client.models.generate_content` /
`genai.Client.aio.models.generate_content` so no network request (and no API
quota) is ever used, and patch `time.sleep` / `asyncio.sleep` so the tests
run instantly instead of waiting through the real backoff delays.
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

from google.genai import errors

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ipeecs_bot.llm_api.llm_model.llm_gemini import GeminiLLMProvider


def make_provider() -> GeminiLLMProvider:
    """Builds a provider with a dummy key. Client construction is offline/lazy,
    so this never touches the network."""
    return GeminiLLMProvider(api_key="fake-key-for-test")


def overloaded_error() -> errors.ServerError:
    return errors.ServerError(code=503, response_json={"message": "high demand", "status": "UNAVAILABLE"})


def ok_response(text: str = "ok") -> MagicMock:
    return MagicMock(text=text)


def test_async_retry_recovers_from_503_with_exponential_backoff():
    provider = make_provider()
    mock_call = AsyncMock(side_effect=[overloaded_error(), overloaded_error(), overloaded_error(), ok_response()])

    async def run():
        with patch.object(provider.client.aio.models, "generate_content", new=mock_call), \
                patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            return await provider.generate_response("問題"), mock_sleep

    result, mock_sleep = asyncio.run(run())
    assert result == "ok"
    assert mock_call.call_count == 4
    assert mock_sleep.await_args_list == [call(1.0), call(2.0), call(4.0)]


def test_async_retry_gives_up_after_6_retries():
    provider = make_provider()
    mock_call = AsyncMock(side_effect=overloaded_error())

    async def run():
        with patch.object(provider.client.aio.models, "generate_content", new=mock_call), \
                patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            try:
                await provider.generate_response("問題")
                assert False, "expected ServerError to propagate"
            except errors.ServerError as e:
                assert e.code == 503
            return mock_sleep

    mock_sleep = asyncio.run(run())
    assert mock_call.call_count == provider.MAX_RETRIES + 1 == 7
    assert [c.args[0] for c in mock_sleep.await_args_list] == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def test_async_does_not_retry_non_503_errors():
    provider = make_provider()
    bad_request = errors.ClientError(code=400, response_json={"message": "bad", "status": "INVALID_ARGUMENT"})
    mock_call = AsyncMock(side_effect=bad_request)

    async def run():
        with patch.object(provider.client.aio.models, "generate_content", new=mock_call), \
                patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            try:
                await provider.generate_response("問題")
                assert False, "expected ClientError to propagate"
            except errors.ClientError:
                pass
            return mock_sleep

    mock_sleep = asyncio.run(run())
    assert mock_call.call_count == 1
    mock_sleep.assert_not_awaited()


def test_sync_retry_recovers_from_503():
    provider = make_provider()
    with patch.object(provider.client.models, "generate_content",
                      side_effect=[overloaded_error(), ok_response()]) as mock_call, \
            patch("time.sleep") as mock_sleep:
        result = provider.generate_response_sync("問題")

    assert result == "ok"
    assert mock_call.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


def test_sync_retry_gives_up_after_6_retries():
    provider = make_provider()
    with patch.object(provider.client.models, "generate_content", side_effect=overloaded_error()) as mock_call, \
            patch("time.sleep"):
        try:
            provider.generate_response_sync("問題")
            assert False, "expected ServerError to propagate"
        except errors.ServerError:
            pass

    assert mock_call.call_count == provider.MAX_RETRIES + 1


if __name__ == "__main__":
    tests = [
        test_async_retry_recovers_from_503_with_exponential_backoff,
        test_async_retry_gives_up_after_6_retries,
        test_async_does_not_retry_non_503_errors,
        test_sync_retry_recovers_from_503,
        test_sync_retry_gives_up_after_6_retries,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("All LLM retry tests passed (no API quota used).")

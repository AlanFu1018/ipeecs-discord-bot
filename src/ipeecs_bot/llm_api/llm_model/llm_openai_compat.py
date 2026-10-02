"""Shared LLM provider for OpenAI-compatible APIs (Groq, NVIDIA NIM, ...) using the openai SDK."""
import asyncio
import time
from typing import Optional
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAI

from ..llm_base import BaseLLMProvider
from ...core.logger import logger


class OpenAICompatibleLLMProvider(BaseLLMProvider):
    """Base provider for OpenAI-compatible chat completion APIs with exponential-backoff retry.

    Subclasses only need to set PROVIDER_NAME, BASE_URL and DEFAULT_MODEL.
    """

    PROVIDER_NAME = "OpenAI-compatible"
    API_KEY_ENV = "API_KEY"
    BASE_URL: Optional[str] = None
    DEFAULT_MODEL = ""

    # Retries after the first attempt, so a request makes at most MAX_RETRIES + 1 calls.
    # Worst-case total wait: 1 + 2 + 4 + 8 + 16 + 32 = 63s.
    MAX_RETRIES = 6
    INITIAL_DELAY = 1.0
    BACKOFF_MULTIPLIER = 2.0
    # 429 (rate limit) is common on Groq / NIM free tiers; 5xx means the server is overloaded or flaky.
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(self, api_key: str, default_model: Optional[str] = None, base_url: Optional[str] = None):
        if not api_key:
            raise ValueError(f"{self.API_KEY_ENV} is not set or empty.")
        self.api_key = api_key
        self.default_model = default_model or self.DEFAULT_MODEL
        self.base_url = base_url or self.BASE_URL
        # Disable the SDK's built-in retries so our own backoff is the only retry layer.
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)
        self.aclient = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)

    @classmethod
    def _is_retryable(cls, e: Exception) -> bool:
        """Rate-limit / server-side / connection errors are worth retrying; other errors are not."""
        if isinstance(e, APIStatusError):
            return e.status_code in cls.RETRYABLE_STATUS_CODES
        return isinstance(e, APIConnectionError)

    def _retry_message(self, e: Exception, delay: float, retry: int) -> str:
        status = getattr(e, "status_code", "connection error")
        return (
            f"{self.PROVIDER_NAME} {status}. Retrying in {delay:.1f}s "
            f"(retry {retry + 1}/{self.MAX_RETRIES})..."
        )

    async def _call_with_retry_async(self, func, *args, **kwargs):
        """Retries an async API call on retryable errors with exponential backoff."""
        delay = self.INITIAL_DELAY
        for retry in range(self.MAX_RETRIES + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if not self._is_retryable(e) or retry == self.MAX_RETRIES:
                    raise
                logger.warning(self._retry_message(e, delay, retry))
                await asyncio.sleep(delay)
                delay *= self.BACKOFF_MULTIPLIER

    def _call_with_retry(self, func, *args, **kwargs):
        """Retries a sync API call on retryable errors with exponential backoff."""
        delay = self.INITIAL_DELAY
        for retry in range(self.MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not self._is_retryable(e) or retry == self.MAX_RETRIES:
                    raise
                logger.warning(self._retry_message(e, delay, retry))
                time.sleep(delay)
                delay *= self.BACKOFF_MULTIPLIER

    @staticmethod
    def _build_messages(prompt: str, system_instruction: Optional[str]) -> list:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _extract_text(response) -> str:
        if response and response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = 0.2,
        max_output_tokens: Optional[int] = 1500,
    ) -> str:
        """Asynchronously generates text response via the chat completions API."""
        try:
            response = await self._call_with_retry_async(
                self.aclient.chat.completions.create,
                model=self.default_model,
                messages=self._build_messages(prompt, system_instruction),
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
            return self._extract_text(response)
        except Exception as e:
            logger.error(f"{self.PROVIDER_NAME} generation error: {e}", exc_info=True)
            raise e

    def generate_response_sync(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = 0.2,
        max_output_tokens: Optional[int] = 1500,
    ) -> str:
        """Synchronously generates text response via the chat completions API."""
        try:
            response = self._call_with_retry(
                self.client.chat.completions.create,
                model=self.default_model,
                messages=self._build_messages(prompt, system_instruction),
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
            return self._extract_text(response)
        except Exception as e:
            logger.error(f"{self.PROVIDER_NAME} sync generation error: {e}", exc_info=True)
            raise e

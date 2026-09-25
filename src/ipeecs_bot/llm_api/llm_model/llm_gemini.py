"""Google Gemini LLM provider implementation using google-genai SDK."""
import asyncio
import time
from typing import Optional
from google import genai
from google.genai import errors, types

from ..llm_base import BaseLLMProvider
from ...core.logger import logger


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini LLM Provider with exponential-backoff retry on 503 (model overloaded)."""

    # Retries after the first attempt, so a request makes at most MAX_RETRIES + 1 calls.
    # Worst-case total wait: 1 + 2 + 4 + 8 + 16 + 32 = 63s.
    MAX_RETRIES = 6
    INITIAL_DELAY = 1.0
    BACKOFF_MULTIPLIER = 2.0

    def __init__(self, api_key: str, default_model: str = "gemini-2.0-flash"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set or empty.")
        self.api_key = api_key
        self.default_model = default_model
        self.client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _is_overloaded(e: Exception) -> bool:
        """503 UNAVAILABLE is Gemini's 'high demand' error and is worth retrying; other errors are not."""
        return isinstance(e, errors.ServerError) and e.code == 503

    async def _call_with_retry_async(self, func, *args, **kwargs):
        """Retries an async API call on 503 with exponential backoff, up to MAX_RETRIES times."""
        delay = self.INITIAL_DELAY
        for retry in range(self.MAX_RETRIES + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if not self._is_overloaded(e) or retry == self.MAX_RETRIES:
                    raise
                logger.warning(
                    f"Gemini 503 (model overloaded). Retrying in {delay:.1f}s (retry {retry + 1}/{self.MAX_RETRIES})..."
                )
                await asyncio.sleep(delay)
                delay *= self.BACKOFF_MULTIPLIER

    def _call_with_retry(self, func, *args, **kwargs):
        """Retries a sync API call on 503 with exponential backoff, up to MAX_RETRIES times."""
        delay = self.INITIAL_DELAY
        for retry in range(self.MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not self._is_overloaded(e) or retry == self.MAX_RETRIES:
                    raise
                logger.warning(
                    f"Gemini 503 (model overloaded). Retrying in {delay:.1f}s (retry {retry + 1}/{self.MAX_RETRIES})..."
                )
                time.sleep(delay)
                delay *= self.BACKOFF_MULTIPLIER

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = 0.2,
        max_output_tokens: Optional[int] = 1500,
    ) -> str:
        """Asynchronously generates text response using Gemini API."""
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_instruction,
            )
            response = await self._call_with_retry_async(
                self.client.aio.models.generate_content,
                model=self.default_model,
                contents=prompt,
                config=config,
            )
            if response and response.text:
                return response.text.strip()
            return ""
        except Exception as e:
            logger.error(f"Gemini generation error: {e}", exc_info=True)
            raise e

    def generate_response_sync(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = 0.2,
        max_output_tokens: Optional[int] = 1500,
    ) -> str:
        """Synchronously generates text response using Gemini API."""
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_instruction,
            )
            response = self._call_with_retry(
                self.client.models.generate_content,
                model=self.default_model,
                contents=prompt,
                config=config,
            )
            if response and response.text:
                return response.text.strip()
            return ""
        except Exception as e:
            logger.error(f"Gemini sync generation error: {e}", exc_info=True)
            raise e

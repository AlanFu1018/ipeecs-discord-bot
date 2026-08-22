"""Google Gemini LLM provider implementation using google-genai SDK."""
from typing import Optional
from google import genai
from google.genai import types

from ..llm_base import BaseLLMProvider
from ...core.logger import logger


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini LLM Provider."""

    def __init__(self, api_key: str, default_model: str = "gemini-2.0-flash"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set or empty.")
        self.api_key = api_key
        self.default_model = default_model
        self.client = genai.Client(api_key=self.api_key)

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
            response = await self.client.aio.models.generate_content(
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
            response = self.client.models.generate_content(
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

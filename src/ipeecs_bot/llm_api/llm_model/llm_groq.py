"""Groq LLM provider implementation (OpenAI-compatible API)."""
from .llm_openai_compat import OpenAICompatibleLLMProvider


class GroqLLMProvider(OpenAICompatibleLLMProvider):
    """Groq LLM Provider. Model list: https://console.groq.com/docs/models"""

    PROVIDER_NAME = "Groq"
    API_KEY_ENV = "GROQ_API_KEY"
    BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

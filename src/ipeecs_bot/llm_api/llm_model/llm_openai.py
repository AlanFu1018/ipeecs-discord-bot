"""OpenAI (GPT) LLM provider implementation."""
from .llm_openai_compat import OpenAICompatibleLLMProvider


class OpenAILLMProvider(OpenAICompatibleLLMProvider):
    """OpenAI LLM Provider. Model list: https://platform.openai.com/docs/models

    Pass base_url to use an OpenAI-compatible proxy instead of the official endpoint.
    """

    PROVIDER_NAME = "OpenAI"
    API_KEY_ENV = "OPENAI_API_KEY"
    BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini"

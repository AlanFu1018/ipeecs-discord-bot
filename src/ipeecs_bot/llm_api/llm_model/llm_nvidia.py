"""NVIDIA NIM LLM provider implementation (OpenAI-compatible API)."""
from .llm_openai_compat import OpenAICompatibleLLMProvider


class NvidiaLLMProvider(OpenAICompatibleLLMProvider):
    """NVIDIA NIM LLM Provider. Model list: https://build.nvidia.com/models

    BASE_URL points to NVIDIA's hosted API; pass base_url to use a self-hosted NIM instead.
    """

    PROVIDER_NAME = "NVIDIA NIM"
    API_KEY_ENV = "NVIDIA_API_KEY"
    BASE_URL = "https://integrate.api.nvidia.com/v1"
    DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"

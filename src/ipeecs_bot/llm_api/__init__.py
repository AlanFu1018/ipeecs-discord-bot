"""LLM & Embedding abstractions and factory module."""
from .llm_base import BaseLLMProvider
from .embed_base import BaseEmbeddingProvider
from .llm_model.llm_gemini import GeminiLLMProvider
from .llm_model.llm_groq import GroqLLMProvider
from .llm_model.llm_nvidia import NvidiaLLMProvider
from .embed_method.embed_gemini import GeminiEmbeddingProvider
from .embed_method.embed_local import LocalEmbeddingProvider
from ..core.config import Settings


def get_llm_provider(settings: Settings) -> BaseLLMProvider:
    """Factory function returning the configured LLM provider."""
    provider_name = settings.llm_provider.lower()
    if provider_name == "gemini":
        return GeminiLLMProvider(
            api_key=settings.gemini_api_key,
            default_model=settings.llm_model,
        )
    elif provider_name == "groq":
        return GroqLLMProvider(
            api_key=settings.groq_api_key,
            default_model=settings.llm_model,
        )
    elif provider_name in ("nvidia", "nim"):
        return NvidiaLLMProvider(
            api_key=settings.nvidia_api_key,
            default_model=settings.llm_model,
            base_url=settings.llm_base_url,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def get_embedding_provider(settings: Settings) -> BaseEmbeddingProvider:
    """Factory function returning the configured Embedding provider."""
    provider_name = settings.embed_provider.lower()
    if provider_name == "gemini":
        return GeminiEmbeddingProvider(
            api_key=settings.gemini_api_key,
            model_name=settings.embed_model,
            batch_size=settings.embed_batch_size,
            delay_seconds=settings.embed_delay_seconds,
        )
    elif provider_name == "local":
        return LocalEmbeddingProvider(model_name=settings.embed_model)
    else:
        raise ValueError(f"Unsupported Embedding provider: {settings.embed_provider}")


__all__ = [
    "BaseLLMProvider",
    "BaseEmbeddingProvider",
    "GeminiLLMProvider",
    "GroqLLMProvider",
    "NvidiaLLMProvider",
    "GeminiEmbeddingProvider",
    "LocalEmbeddingProvider",
    "get_llm_provider",
    "get_embedding_provider",
]

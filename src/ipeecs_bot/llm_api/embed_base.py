"""Abstract base class for Embedding providers."""
from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Abstract interface for embedding generation."""

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Embeds a single query string asynchronously."""
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds multiple documents/chunks asynchronously."""
        pass

    @abstractmethod
    def embed_query_sync(self, text: str) -> List[float]:
        """Embeds a single query string synchronously."""
        pass

    @abstractmethod
    def embed_documents_sync(self, texts: List[str]) -> List[List[float]]:
        """Embeds multiple documents/chunks synchronously."""
        pass

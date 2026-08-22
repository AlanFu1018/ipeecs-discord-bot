"""Local Embedding provider (SentenceTransformer fallback adapter)."""
from typing import List
from ..embed_base import BaseEmbeddingProvider
from ...core.logger import logger


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local SentenceTransformer embedding provider for offline/custom embeddings."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            logger.warning(
                "sentence_transformers is not installed. "
                "Please install sentence-transformers to use LocalEmbeddingProvider."
            )
            self.model = None

    async def embed_query(self, text: str) -> List[float]:
        return self.embed_query_sync(text)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents_sync(texts)

    def embed_query_sync(self, text: str) -> List[float]:
        if self.model is None:
            raise RuntimeError("sentence_transformers is not available.")
        return self.model.encode(text).tolist()

    def embed_documents_sync(self, texts: List[str]) -> List[List[float]]:
        if self.model is None:
            raise RuntimeError("sentence_transformers is not available.")
        return self.model.encode(texts).tolist()

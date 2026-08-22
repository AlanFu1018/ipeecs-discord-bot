"""Google Gemini Embedding provider implementation using google-genai SDK."""
import time
from typing import List
from google import genai
from google.genai import errors

from ..embed_base import BaseEmbeddingProvider
from ...core.logger import logger


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using Google Gemini embedding models with automatic retry and rate-limiting."""

    def __init__(self, api_key: str, model_name: str = "gemini-embedding-001"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set or empty.")
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)

    def _call_with_retry(self, func, *args, max_retries: int = 6, initial_delay: float = 10.0, **kwargs):
        """Helper to retry API calls on rate limits (429) or transient errors."""
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except errors.ClientError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    wait_time = max(delay, 20.0)  # Wait 20s+ when rate limited to reset quota window
                    logger.warning(f"Rate limit (429) hit on embedding call. Waiting {wait_time:.1f}s before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    delay *= 1.5
                else:
                    raise e
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Transient embedding error: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= 1.5
                else:
                    raise e

    async def embed_query(self, text: str) -> List[float]:
        """Embeds a single query string asynchronously."""
        import asyncio
        delay = 5.0
        for attempt in range(5):
            try:
                response = await self.client.aio.models.embed_content(
                    model=self.model_name,
                    contents=text,
                )
                if response.embeddings and len(response.embeddings) > 0:
                    return response.embeddings[0].values or []
                return []
            except errors.ClientError as e:
                if e.code == 429 and attempt < 4:
                    wait = max(delay, 15.0)
                    logger.warning(f"Rate limit on query embed. Waiting {wait:.1f}s...")
                    await asyncio.sleep(wait)
                    delay *= 1.5
                else:
                    logger.error(f"Gemini embed_query error: {e}")
                    raise e
            except Exception as e:
                logger.error(f"Gemini embed_query error: {e}")
                raise e
        return []

    async def embed_documents(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Embeds multiple documents asynchronously with batching and rate pacing."""
        import asyncio
        if not texts:
            return []
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(6):
                try:
                    response = await self.client.aio.models.embed_content(
                        model=self.model_name,
                        contents=batch,
                    )
                    if response.embeddings:
                        embeddings.extend([emb.values or [] for emb in response.embeddings])
                    break
                except errors.ClientError as e:
                    if e.code == 429 and attempt < 5:
                        wait = max(5.0 * (attempt + 1), 20.0)
                        logger.warning(f"Rate limit hit during embed_documents. Sleeping {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        raise e
            await asyncio.sleep(4.0)
        return embeddings

    def embed_query_sync(self, text: str) -> List[float]:
        """Embeds a single query string synchronously."""
        response = self._call_with_retry(
            self.client.models.embed_content,
            model=self.model_name,
            contents=text,
        )
        if response.embeddings and len(response.embeddings) > 0:
            return response.embeddings[0].values or []
        return []

    def embed_documents_sync(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Embeds multiple documents synchronously with batching and rate pacing."""
        if not texts:
            return []
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._call_with_retry(
                self.client.models.embed_content,
                model=self.model_name,
                contents=batch,
            )
            if response.embeddings:
                embeddings.extend([emb.values or [] for emb in response.embeddings])
            # Sleep 4s between batches so we stay under 15 requests per minute
            time.sleep(4.0)
        return embeddings

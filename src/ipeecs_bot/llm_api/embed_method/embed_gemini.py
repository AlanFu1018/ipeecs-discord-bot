"""Google Gemini Embedding provider implementation using google-genai SDK."""
import time
from typing import List, Optional
from google import genai
from google.genai import errors

from ..embed_base import BaseEmbeddingProvider
from ...core.logger import logger


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using Google Gemini embedding models with automatic retry, batching, and rate-limiting."""

    # Shared retry defaults for every sync/async call site below, so tuning
    # retry behavior only requires touching one place.
    DEFAULT_MAX_RETRIES = 8
    DEFAULT_INITIAL_DELAY = 15.0
    RATE_LIMIT_MIN_WAIT = 50.0  # Wait 50s+ on 429 to reset the quota window
    BACKOFF_MULTIPLIER = 1.5

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-embedding-001",
        batch_size: int = 10,
        delay_seconds: float = 5.0,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set or empty.")
        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds
        self.client = genai.Client(api_key=self.api_key)

    def _call_with_retry(self, func, *args, max_retries: Optional[int] = None, initial_delay: Optional[float] = None, **kwargs):
        """Helper to retry sync API calls on rate limits (429) or transient errors with exponential backoff."""
        max_retries = max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        delay = initial_delay if initial_delay is not None else self.DEFAULT_INITIAL_DELAY
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except errors.ClientError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    wait_time = max(delay, self.RATE_LIMIT_MIN_WAIT)
                    logger.warning(
                        f"Rate limit (429) hit on embedding call. Waiting {wait_time:.1f}s before retry (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                    delay *= self.BACKOFF_MULTIPLIER
                else:
                    raise e
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Transient embedding error: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= self.BACKOFF_MULTIPLIER
                else:
                    raise e

    async def _call_with_retry_async(self, func, *args, max_retries: Optional[int] = None, initial_delay: Optional[float] = None, **kwargs):
        """Helper to retry async API calls on rate limits (429) or transient errors with exponential backoff."""
        import asyncio
        max_retries = max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        delay = initial_delay if initial_delay is not None else self.DEFAULT_INITIAL_DELAY
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except errors.ClientError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    wait_time = max(delay, self.RATE_LIMIT_MIN_WAIT)
                    logger.warning(
                        f"Rate limit (429) hit on embedding call. Waiting {wait_time:.1f}s before retry (attempt {attempt + 1}/{max_retries})..."
                    )
                    await asyncio.sleep(wait_time)
                    delay *= self.BACKOFF_MULTIPLIER
                else:
                    raise e
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Transient embedding error: {e}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    delay *= self.BACKOFF_MULTIPLIER
                else:
                    raise e

    async def embed_query(self, text: str) -> List[float]:
        """Embeds a single query string asynchronously."""
        response = await self._call_with_retry_async(
            self.client.aio.models.embed_content,
            model=self.model_name,
            contents=text,
        )
        if response.embeddings and len(response.embeddings) > 0:
            return response.embeddings[0].values or []
        return []

    async def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Embeds multiple documents asynchronously with batching and rate pacing."""
        import asyncio
        if not texts:
            return []

        effective_batch_size = batch_size or self.batch_size
        embeddings = []
        total_batches = (len(texts) + effective_batch_size - 1) // effective_batch_size

        for batch_idx, i in enumerate(range(0, len(texts), effective_batch_size), 1):
            batch = texts[i : i + effective_batch_size]
            response = await self._call_with_retry_async(
                self.client.aio.models.embed_content,
                model=self.model_name,
                contents=batch,
            )
            if response.embeddings:
                embeddings.extend([emb.values or [] for emb in response.embeddings])

            logger.info(f"[Embedding] Finished batch {batch_idx}/{total_batches} ({len(embeddings)}/{len(texts)} chunks)")
            if i + effective_batch_size < len(texts):
                await asyncio.sleep(self.delay_seconds)

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

    def embed_documents_sync(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Embeds multiple documents synchronously with batching and rate pacing."""
        if not texts:
            return []

        effective_batch_size = batch_size or self.batch_size
        embeddings = []
        total_batches = (len(texts) + effective_batch_size - 1) // effective_batch_size

        for batch_idx, i in enumerate(range(0, len(texts), effective_batch_size), 1):
            batch = texts[i : i + effective_batch_size]
            response = self._call_with_retry(
                self.client.models.embed_content,
                model=self.model_name,
                contents=batch,
            )
            if response.embeddings:
                embeddings.extend([emb.values or [] for emb in response.embeddings])

            logger.info(f"[Embedding] Finished batch {batch_idx}/{total_batches} ({len(embeddings)}/{len(texts)} chunks)")
            if i + effective_batch_size < len(texts):
                time.sleep(self.delay_seconds)

        return embeddings

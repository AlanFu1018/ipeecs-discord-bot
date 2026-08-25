"""ChromaDB vector database storage and similarity search module."""
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from .parser import DocumentChunk
from ..core.logger import logger
from ..llm_api.embed_base import BaseEmbeddingProvider


class VectorStore:
    """Persistent ChromaDB vector store wrapper."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedding_provider: BaseEmbeddingProvider,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Initialized VectorStore at {self.persist_dir} (Collection: {self.collection_name}, Docs: {self.collection.count()})"
        )

    def count(self) -> int:
        """Returns the total number of document chunks stored."""
        return self.collection.count()

    def reset_collection(self) -> None:
        """Clears all existing documents in the current collection."""
        try:
            existing = self.collection.get()
            existing_ids = existing.get("ids", [])
            if existing_ids:
                self.collection.delete(ids=existing_ids)
                logger.info(f"Reset collection by removing {len(existing_ids)} items: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} is already empty.")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")

    def add_chunks_sync(self, chunks: List[DocumentChunk]) -> int:
        """Generates embeddings and stores document chunks synchronously."""
        if not chunks:
            logger.warning("No chunks provided to add.")
            return 0

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk.content for chunk in chunks]
        ids = [f"chunk_{idx}" for idx in range(len(chunks))]
        metadatas = [chunk.metadata for chunk in chunks]

        # embed_documents_sync batches internally and respects rate pacing
        embeddings = self.embedding_provider.embed_documents_sync(texts)

        logger.info(f"Writing {len(chunks)} embeddings into ChromaDB collection '{self.collection_name}'...")
        chroma_batch_size = 100
        for i in range(0, len(chunks), chroma_batch_size):
            end_idx = i + chroma_batch_size
            self.collection.add(
                ids=ids[i:end_idx],
                documents=texts[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
            )

        logger.info(f"Successfully indexed total {len(chunks)} chunks in VectorStore.")
        return len(chunks)

    async def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Generates embeddings and stores document chunks asynchronously."""
        if not chunks:
            logger.warning("No chunks provided to add.")
            return 0

        logger.info(f"Asynchronously generating embeddings for {len(chunks)} chunks...")
        texts = [chunk.content for chunk in chunks]
        ids = [f"chunk_{idx}" for idx in range(len(chunks))]
        metadatas = [chunk.metadata for chunk in chunks]

        embeddings = await self.embedding_provider.embed_documents(texts)

        logger.info(f"Writing {len(chunks)} embeddings into ChromaDB collection '{self.collection_name}'...")
        chroma_batch_size = 100
        for i in range(0, len(chunks), chroma_batch_size):
            end_idx = i + chroma_batch_size
            self.collection.add(
                ids=ids[i:end_idx],
                documents=texts[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
            )

        logger.info(f"Successfully indexed total {len(chunks)} chunks in VectorStore.")
        return len(chunks)

    async def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs vector similarity search asynchronously."""
        query_vector = await self.embedding_provider.embed_query(query)
        if not query_vector:
            logger.warning(f"Could not generate embedding for query: {query}")
            return []

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )

        matched_docs = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

            for doc, meta, dist, chunk_id in zip(docs, metadatas, distances, ids):
                matched_docs.append({
                    "id": chunk_id,
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                })

        return matched_docs

    def search_sync(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs vector similarity search synchronously."""
        query_vector = self.embedding_provider.embed_query_sync(query)
        if not query_vector:
            logger.warning(f"Could not generate embedding for query: {query}")
            return []

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )

        matched_docs = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

            for doc, meta, dist, chunk_id in zip(docs, metadatas, distances, ids):
                matched_docs.append({
                    "id": chunk_id,
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                })

        return matched_docs

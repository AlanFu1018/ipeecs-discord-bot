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
        """Deletes and recreates the current collection."""
        try:
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Reset collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")

    def add_chunks_sync(self, chunks: List[DocumentChunk], batch_size: int = 50) -> int:
        """Generates embeddings and stores document chunks synchronously."""
        if not chunks:
            logger.warning("No chunks provided to add.")
            return 0

        logger.info(f"Embedding and indexing {len(chunks)} chunks...")
        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [chunk.content for chunk in batch]
            ids = [
                f"{chunk.metadata.get('source', 'doc')}_{chunk.metadata.get('chunk_index', idx)}_{i + idx}"
                for idx, chunk in enumerate(batch)
            ]
            metadatas = [chunk.metadata for chunk in batch]

            embeddings = self.embedding_provider.embed_documents_sync(texts)
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total_added += len(batch)
            logger.info(f"Indexed batch {i // batch_size + 1}: {total_added}/{len(chunks)} chunks")

        logger.info(f"Successfully indexed total {total_added} chunks in VectorStore.")
        return total_added

    async def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 50) -> int:
        """Generates embeddings and stores document chunks asynchronously."""
        if not chunks:
            logger.warning("No chunks provided to add.")
            return 0

        logger.info(f"Asynchronously embedding and indexing {len(chunks)} chunks...")
        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [chunk.content for chunk in batch]
            ids = [
                f"{chunk.metadata.get('source', 'doc')}_{chunk.metadata.get('chunk_index', idx)}_{i + idx}"
                for idx, chunk in enumerate(batch)
            ]
            metadatas = [chunk.metadata for chunk in batch]

            embeddings = await self.embedding_provider.embed_documents(texts)
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total_added += len(batch)

        logger.info(f"Successfully indexed total {total_added} chunks.")
        return total_added

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

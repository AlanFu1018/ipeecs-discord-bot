"""Data Synchronization and Vector Store Ingestion Tool.

Executes web crawling, PDF downloads, Gemini multimodal table-to-markdown conversion,
document parsing, text chunking, embedding generation, and local ChromaDB persistence.
"""
import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ipeecs_bot.core.config import get_settings
from src.ipeecs_bot.core.logger import logger
from src.ipeecs_bot.llm_api import get_embedding_provider
from src.ipeecs_bot.rag.parser import DocumentParser
from src.ipeecs_bot.rag.vector_store import VectorStore
from src.ipeecs_bot.services.crawler import DataCrawler


def run_sync(
    skip_crawl: bool = False,
    skip_llm_convert: bool = False,
    skip_converted: bool = False,
    reset_db: bool = True,
):
    """Executes the data synchronization workflow."""
    settings = get_settings()
    logger.info("=== Starting IPEECS Bot Knowledge Base Sync ===")

    # Step 1: Web Crawling, PDF Downloading & Table Conversion via Gemini
    if not skip_crawl:
        logger.info("[Step 1/3] Crawling Department URLs, downloading PDFs & converting tables with Gemini...")
        crawler = DataCrawler(
            raw_dir=settings.raw_dir,
            markdown_dir=settings.markdown_dir,
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.llm_model,
        )
        crawler.crawl_all(
            urls_file=settings.urls_file,
            skip_llm_convert=skip_llm_convert,
            skip_converted=skip_converted,
        )
    else:
        logger.info("[Step 1/3] Skipping crawl as requested.")

    # Step 2: Document Parsing & Text Chunking
    logger.info("[Step 2/3] Parsing documents and chunking text...")
    parser = DocumentParser(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        chunk_mini=settings.chunk_mini
    )
    chunks = parser.parse_directory(
        raw_dir=settings.raw_dir,
        markdown_dir=settings.markdown_dir,
    )

    if not chunks:
        logger.warning("No document chunks were extracted. Please check raw files or urls.txt.")
        return

    logger.info(f"Generated {len(chunks)} total text chunks.")

    # Step 3: Embedding and Vector Database Storage
    logger.info("[Step 3/3] Generating embeddings and indexing into ChromaDB...")
    embedder = get_embedding_provider(settings)
    vector_store = VectorStore(
        persist_dir=settings.chroma_db_dir,
        collection_name=settings.collection_name,
        embedding_provider=embedder,
    )

    if reset_db:
        logger.info("Clearing existing vector collection...")
        vector_store.reset_collection()

    total_added = vector_store.add_chunks_sync(chunks)
    logger.info(f"=== Knowledge Base Sync Completed! Total Chunks: {total_added} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synchronize IPEECS knowledge base data.")
    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        help="Skip web crawling and only re-index existing raw/markdown files.",
    )
    parser.add_argument(
        "--skip-llm-convert",
        action="store_true",
        help="Execute web crawl and PDF download but skip Gemini table-to-markdown conversion.",
    )
    parser.add_argument(
        "--skip-converted",
        action="store_true",
        help="Skip Gemini LLM conversion for table PDFs that already have a corresponding markdown file.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not clear the existing ChromaDB collection before adding chunks.",
    )
    args = parser.parse_args()

    run_sync(
        skip_crawl=args.skip_crawl,
        skip_llm_convert=args.skip_llm_convert,
        skip_converted=args.skip_converted,
        reset_db=not args.no_reset,
    )

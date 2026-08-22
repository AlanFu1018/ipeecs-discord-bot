"""Main entry point for starting the IPEECS Department Advisor Discord Bot."""
import asyncio
import os
import signal
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ipeecs_bot.bot.bot import IpeecsDiscordBot
from src.ipeecs_bot.core.config import get_settings
from src.ipeecs_bot.core.logger import logger
from src.ipeecs_bot.llm_api import get_embedding_provider, get_llm_provider
from src.ipeecs_bot.rag.vector_store import VectorStore
from src.ipeecs_bot.services.chat_service import ChatService


async def main():
    settings = get_settings()

    if not settings.discord_bot_token:
        logger.error("DISCORD_BOT_TOKEN is not configured in .env file! Exiting...")
        sys.exit(1)

    if not settings.gemini_api_key:
        logger.error("GEMINI_API_KEY is not configured in .env file! Exiting...")
        sys.exit(1)

    logger.info("Initializing IPEECS Discord Bot services...")

    # Initialize Adapters
    llm_provider = get_llm_provider(settings)
    embedding_provider = get_embedding_provider(settings)

    # Initialize Vector Store
    vector_store = VectorStore(
        persist_dir=settings.chroma_db_dir,
        collection_name=settings.collection_name,
        embedding_provider=embedding_provider,
    )

    # Check if vector store has data
    doc_count = vector_store.count()
    if doc_count == 0:
        logger.warning(
            "Vector database is currently empty! "
            "Please run 'python sync_data.py' to crawl and index regulation documents."
        )
    else:
        logger.info(f"Vector store loaded with {doc_count} document chunks.")

    # Initialize Chat Service
    chat_service = ChatService(
        settings=settings,
        llm_provider=llm_provider,
        vector_store=vector_store,
    )

    # Initialize Discord Bot
    bot = IpeecsDiscordBot(
        settings=settings,
        chat_service=chat_service,
    )

    try:
        await bot.start(settings.discord_bot_token)
    except KeyboardInterrupt:
        logger.info("Shutdown requested via KeyboardInterrupt...")
    except Exception as e:
        logger.error(f"Unexpected error while running bot: {e}", exc_info=True)
    finally:
        if not bot.is_closed():
            logger.info("Closing bot connection gracefully...")
            await bot.close()
        logger.info("Bot shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process terminated by user.")

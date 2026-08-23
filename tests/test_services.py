"""Unit and integration tests for IPEECS Discord Bot."""
import asyncio
import sys
import time
from pathlib import Path

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ipeecs_bot.core.config import get_settings
from src.ipeecs_bot.llm_api import get_embedding_provider, get_llm_provider
from src.ipeecs_bot.rag.vector_store import VectorStore
from src.ipeecs_bot.services.chat_service import ChatService
from src.ipeecs_bot.services.session import SessionManager, UserSession


def test_session_sliding_window_and_timeout():
    """Tests session expiration and message limit sliding window."""
    manager = SessionManager(timeout_minutes=1, max_history_turns=2)
    session = manager.get_or_create_session("user_123")

    # Add 6 messages (3 turns) -> should keep only last 4 messages (2 turns)
    session.add_message("user", "Q1", max_turns=2)
    session.add_message("model", "A1", max_turns=2)
    session.add_message("user", "Q2", max_turns=2)
    session.add_message("model", "A2", max_turns=2)
    session.add_message("user", "Q3", max_turns=2)
    session.add_message("model", "A3", max_turns=2)

    assert len(session.messages) == 4
    assert session.messages[0].content == "Q2"
    assert session.messages[-1].content == "A3"

    # Simulate timeout
    session.last_activity = time.time() - 70
    reloaded_session = manager.get_or_create_session("user_123")
    assert len(reloaded_session.messages) == 0  # Should be cleared


async def test_llm_query_rewriting_and_fallback():
    """Tests query rewriting, fallback response, and cross-page table retrieval."""
    settings = get_settings()
    llm = get_llm_provider(settings)
    embedding = get_embedding_provider(settings)
    vector_store = VectorStore(settings.chroma_db_dir, settings.collection_name, embedding)
    chat_service = ChatService(settings, llm, vector_store)

    # Test reset command
    reset_reply = await chat_service.answer_message("test_user_001", "/reset")
    assert "重置" in reset_reply or "reset" in reset_reply.lower()

    # Test query rewriting directly
    session = chat_service.session_manager.get_or_create_session("test_user_002")
    session.add_message("user", "請問資電學士班大二必修有哪些？")
    session.add_message("model", "大二必修包含資料結構、演算法等。")
    rewritten = await chat_service.rewrite_query(session, "那大三呢？")
    print("Rewritten query result:", rewritten)
    assert len(rewritten) > 0

    # Test entrepreneurship course query (table retrieval test)
    answer = await chat_service.answer_message(
        "test_user_003",
        "請問「創意與創業」學分學程中，創業專題類別包含哪些課程？至少需要修幾學分？"
    )
    print("Entrepreneurship test response:\n", answer)
    assert "4" in answer or "四" in answer
    assert "專題" in answer


if __name__ == "__main__":
    print("Running session unit tests...")
    test_session_sliding_window_and_timeout()
    print("Session tests passed!")

    print("Running LLM async tests...")
    asyncio.run(test_llm_query_rewriting_and_fallback())
    print("All tests passed successfully!")

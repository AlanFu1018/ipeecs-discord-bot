"""Session and conversation history management."""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.logger import logger


@dataclass
class ChatMessage:
    """A single chat message in a session."""
    role: str  # "user" or "model"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class UserSession:
    """User conversation session containing message history and last activity time."""
    user_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)

    def is_expired(self, timeout_seconds: float) -> bool:
        """Checks if session has timed out due to inactivity."""
        return (time.time() - self.last_activity) > timeout_seconds

    def add_message(self, role: str, content: str, max_turns: int = 5) -> None:
        """Appends a message and keeps only the latest max_turns (user+model pairs)."""
        self.messages.append(ChatMessage(role=role, content=content, timestamp=time.time()))
        self.last_activity = time.time()
        # Keep latest (max_turns * 2) messages
        if len(self.messages) > max_turns * 2:
            self.messages = self.messages[-(max_turns * 2) :]

    def clear(self) -> None:
        """Clears all conversation history."""
        self.messages.clear()
        self.last_activity = time.time()

    def get_history_summary(self) -> str:
        """Formats conversation history for LLM query rewriting."""
        history_lines = []
        for msg in self.messages:
            role_label = "使用者" if msg.role == "user" else "顧問機器人"
            history_lines.append(f"{role_label}: {msg.content}")
        return "\n".join(history_lines)


class SessionManager:
    """Manages active user sessions in memory."""

    def __init__(self, timeout_minutes: int = 15, max_history_turns: int = 5):
        self.timeout_seconds = timeout_minutes * 60
        self.max_history_turns = max_history_turns
        self.sessions: Dict[str, UserSession] = {}

    def get_or_create_session(self, user_id: str) -> UserSession:
        """Retrieves or creates a session for the given user ID."""
        session = self.sessions.get(user_id)
        if session is None:
            session = UserSession(user_id=user_id)
            self.sessions[user_id] = session
            logger.info(f"Created new session for user {user_id}")
        elif session.is_expired(self.timeout_seconds):
            session.clear()
            logger.info(f"Session for user {user_id} expired after {self.timeout_seconds}s and was reset.")
        return session

    def reset_session(self, user_id: str) -> bool:
        """Explicitly resets a user's session."""
        if user_id in self.sessions:
            self.sessions[user_id].clear()
            logger.info(f"Explicitly reset session for user {user_id}")
            return True
        return False

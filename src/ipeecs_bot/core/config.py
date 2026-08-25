"""Configuration loader combining YAML config and .env variables."""
import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Load environment variables from .env
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings:
    """Application settings and runtime parameters."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.base_dir = BASE_DIR
        self.config_path = self.base_dir / config_path
        self._raw_config = self._load_yaml()

        # Environment keys
        self.discord_bot_token = (
            os.getenv("DISCORD_BOT_TOKEN")
            or os.getenv("DISCORD_TOKEN")
            or ""
        )
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or ""

        # Bot configs
        bot_cfg = self._raw_config.get("bot", {})
        self.command_prefix = bot_cfg.get("command_prefix", "!")
        self.session_timeout_minutes = int(bot_cfg.get("session_timeout_minutes", 15))
        self.max_history_turns = int(bot_cfg.get("max_history_turns", 5))

        # LLM configs
        llm_cfg = self._raw_config.get("llm", {})
        self.llm_provider = llm_cfg.get("provider", "gemini")
        self.llm_model = llm_cfg.get("model", "gemini-3.1-flash-lite")
        self.llm_temperature = float(llm_cfg.get("temperature", 0.2))
        self.llm_max_output_tokens = int(llm_cfg.get("max_output_tokens", 1500))

        # Embedding configs
        embed_cfg = self._raw_config.get("embedding", {})
        self.embed_provider = embed_cfg.get("provider", "gemini")
        self.embed_model = embed_cfg.get("model", "gemini-embedding-001")
        self.embed_dimension = int(embed_cfg.get("dimension", 3072))
        self.embed_batch_size = int(embed_cfg.get("batch_size", 10))
        self.embed_delay_seconds = float(embed_cfg.get("delay_seconds", 5.0))

        # RAG configs
        rag_cfg = self._raw_config.get("rag", {})
        self.top_k = int(rag_cfg.get("top_k", 9))
        self.chunk_size = int(rag_cfg.get("chunk_size", 600))
        self.chunk_mini = int(rag_cfg.get("chunk_mini", 100))
        self.chunk_overlap = int(rag_cfg.get("chunk_overlap", 100))
        self.collection_name = rag_cfg.get("collection_name", "ipeecs_knowledge_base")

        # Paths
        paths_cfg = self._raw_config.get("paths", {})
        self.urls_file = self.base_dir / paths_cfg.get("urls_file", "config/urls.txt")
        self.raw_dir = self.base_dir / paths_cfg.get("raw_dir", "res/data/raw")
        self.markdown_dir = self.base_dir / paths_cfg.get("markdown_dir", "res/data/markdown")
        self.chroma_db_dir = self.base_dir / paths_cfg.get("chroma_db_dir", "res/data/chroma_db")

        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)

        # Department contact info fallback
        self.department_info = self._raw_config.get(
            "department_info",
            {
                "name": "資訊電機學院學士班辦公室",
                "phone": "03-4227151 分機 35007",
                "email": "ncu35007@ncu.edu.tw",
                "location": "工程五館E6 B棟106室 (E6-B106)",
                "office_hours": "週一至週五 08:30 - 17:00",
            },
        )

    def _load_yaml(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}


_settings_instance = None


def get_settings() -> Settings:
    """Returns the singleton Settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance

from typing import Any, Dict
import yaml
from pathlib import Path

def load_yaml(file_path: Path) -> Dict[str, Any]:
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

dic = load_yaml(Path("C:\ALL FILES\Code\ipeecs_discord_bot\ipeecs-discord-bot\config\config.yaml"))
print(dic["llm"]["model"])
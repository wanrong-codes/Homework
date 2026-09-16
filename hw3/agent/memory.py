from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


class MemoryStore:
    def __init__(self, path: str = "memory_store.json"):
        self.path = path
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def add_preference(self, user_id: str, key: str, value: Any) -> None:
        user_mem = self._data.setdefault(user_id, {})
        user_mem[key] = {"value": value, "updated_at": time.time()}
        self._save()

    def get_preferences(self, user_id: str) -> Dict[str, Any]:
        return {k: v["value"] for k, v in self._data.get(user_id, {}).items()}

    def get_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        return self._data.get(user_id, {}).get(key, {}).get("value", default)

    def forget(self, user_id: str, key: Optional[str] = None) -> None:
        if user_id not in self._data:
            return
        if key is None:
            del self._data[user_id]
        else:
            self._data[user_id].pop(key, None)
        self._save()

    def format_for_prompt(self, user_id: str) -> str:
        prefs = self.get_preferences(user_id)
        if not prefs:
            return ""
        lines = [f"- {k}: {v}" for k, v in prefs.items()]
        return "已知的用户偏好（回答/规划时请纳入考虑）：\n" + "\n".join(lines)

    def inject_into_messages(self, user_id: str, messages: List[dict]) -> List[dict]:
        memory_text = self.format_for_prompt(user_id)
        if not memory_text:
            return list(messages)
        return [{"role": "system", "content": memory_text}, *messages]

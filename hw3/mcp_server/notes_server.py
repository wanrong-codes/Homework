from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("notes-server")

_STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notes_store.json")


def _load() -> list[str]:
    if os.path.exists(_STORE_PATH):
        with open(_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save(notes: list[str]) -> None:
    with open(_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


@mcp.tool()
def add_note(text: str) -> str:
    notes = _load()
    notes.append(text)
    _save(notes)
    return f"已记录笔记（当前共 {len(notes)} 条）：{text}"


@mcp.tool()
def list_notes() -> str:
    notes = _load()
    if not notes:
        return "目前没有任何笔记。"
    return "\n".join(f"{i + 1}. {n}" for i, n in enumerate(notes))


@mcp.tool()
def clear_notes() -> str:
    _save([])
    return "已清空所有笔记。"


if __name__ == "__main__":
    mcp.run()

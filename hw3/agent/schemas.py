from __future__ import annotations

import json
from typing import List

from .tools import Tool


def to_function_calling_schemas(tools: List[Tool]) -> List[dict]:
    return [t.to_openai_schema() for t in tools]


def to_react_tool_descriptions(tools: List[Tool]) -> str:
    lines = []
    for t in tools:
        params = json.dumps(t.parameters.get("properties", {}), ensure_ascii=False)
        lines.append(f"- {t.name}: {t.description}\n  参数(JSON Schema properties): {params}")
    return "\n".join(lines)

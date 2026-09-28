import hashlib
import json
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

import jsonschema
from fastapi import FastAPI, Request, Response

app = FastAPI(title="hw4 MCP-style Tool Server")


# ------------------------------ 工具实现 ------------------------------ #
_WEATHER = {
    "北京": ("晴", 22), "上海": ("多云", 26), "广州": ("雷阵雨", 31), "深圳": ("多云", 30),
    "杭州": ("小雨", 24), "成都": ("阴", 21), "beijing": ("晴", 22), "shanghai": ("多云", 26), "tokyo": ("晴", 25),
}


def get_weather(city: str) -> dict:
    key = city.strip().lower()
    if key in _WEATHER:
        cond, temp = _WEATHER[key]
    else:
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        cond, temp = ["晴", "多云", "阴", "小雨"][h % 4], 10 + h % 20
    return {"city": city, "condition": cond, "temperature_c": temp}


def get_time(timezone: str = "Asia/Shanghai") -> dict:
    now = datetime.now(ZoneInfo(timezone))
    return {"timezone": timezone, "iso": now.isoformat(timespec="seconds"),
            "weekday": now.strftime("%A")}


def text_stats(text: str) -> dict:
    return {"chars": len(text), "words": len(text.split()), "lines": text.count("\n") + 1}


TOOLS: dict[str, dict[str, Any]] = {
    "get_weather": {
        "description": "查询城市当前天气（演示数据）。",
        "inputSchema": {"type": "object",
                        "properties": {"city": {"type": "string", "description": "城市名，如 北京"}},
                        "required": ["city"], "additionalProperties": False},
        "handler": get_weather,
    },
    "get_time": {
        "description": "获取指定时区的当前时间。",
        "inputSchema": {"type": "object",
                        "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai",
                                                    "description": "IANA 时区，如 Asia/Shanghai"}},
                        "additionalProperties": False},
        "handler": get_time,
    },
    "text_stats": {
        "description": "统计文本的字符数、单词数、行数。",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                        "required": ["text"], "additionalProperties": False},
        "handler": text_stats,
    },
}


# ------------------------------ JSON-RPC ------------------------------ #
def _ok(rid, result):
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid, code, message):
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def _list_tools() -> list[dict]:
    return [{"name": n, "description": t["description"], "inputSchema": t["inputSchema"]}
            for n, t in TOOLS.items()]


def _call_tool(name: str, arguments: dict) -> dict:
    tool = TOOLS[name]
    try:
        jsonschema.validate(arguments, tool["inputSchema"])
        handler: Callable[..., Any] = tool["handler"]
        out = handler(**arguments)
        return {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}],
                "structuredContent": out, "isError": False}
    except Exception as e:  # 工具执行错误按 MCP 约定放在 result 里（isError=true），而不是 JSON-RPC error
        return {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return _err(None, -32700, "Parse error")
    rid, method, params = body.get("id"), body.get("method"), body.get("params") or {}

    if method and method.startswith("notifications/"):
        return Response(status_code=202)  # 通知没有响应体
    if method == "initialize":
        return _ok(rid, {"protocolVersion": "2025-03-26", "capabilities": {"tools": {"listChanged": False}},
                         "serverInfo": {"name": "hw4-mcp-server", "version": "0.1.0"}})
    if method == "tools/list":
        return _ok(rid, {"tools": _list_tools()})
    if method == "tools/call":
        name = params.get("name")
        if name not in TOOLS:
            return _err(rid, -32602, f"Unknown tool: {name}")
        return _ok(rid, _call_tool(name, params.get("arguments") or {}))
    return _err(rid, -32601, f"Method not found: {method}")


@app.get("/tools")
async def list_tools_rest():
    return {"tools": _list_tools()}


@app.get("/health")
async def health():
    return {"status": "ok", "tools": list(TOOLS)}

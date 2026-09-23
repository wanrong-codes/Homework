from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

from .tools import Tool

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MCP_AVAILABLE = False


@dataclass
class MCPToolInfo:
    name: str
    description: str
    input_schema: Dict[str, Any]


def get_default_mcp_server_params() -> "StdioServerParameters":
   
    if not _MCP_AVAILABLE:
        raise RuntimeError("请先 `pip install mcp` 才能使用 MCP 功能。")

    command = os.environ.get("MCP_SERVER_COMMAND")
    args_env = os.environ.get("MCP_SERVER_ARGS")

    if command:
        args = args_env.split(",") if args_env else []
        return StdioServerParameters(command=command, args=args)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_script = os.path.join(project_root, "mcp_server", "notes_server.py")
    return StdioServerParameters(command=sys.executable, args=[server_script])


async def _list_tools_async(server_params: "StdioServerParameters") -> List[MCPToolInfo]:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            return [
                MCPToolInfo(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema or {"type": "object", "properties": {}},
                )
                for t in resp.tools
            ]


async def _call_tool_async(server_params: "StdioServerParameters", name: str, arguments: Dict[str, Any]) -> str:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments=arguments)
            parts = []
            for block in result.content:
                text = getattr(block, "text", None)
                parts.append(text if text is not None else str(block))
            return "\n".join(parts) if parts else "(MCP 工具没有返回内容)"


class MCPToolProxy(Tool):

    def __init__(self, info: MCPToolInfo, server_params: "StdioServerParameters"):
        self.name = f"mcp_{info.name}"
        self.description = f"[通过MCP提供] {info.description}"
        self.parameters = info.input_schema or {"type": "object", "properties": {}}
        self._remote_name = info.name
        self._server_params = server_params

    def run(self, **kwargs: Any) -> str:
        try:
            return asyncio.run(_call_tool_async(self._server_params, self._remote_name, kwargs))
        except Exception as exc:  # noqa: BLE001
            return f"调用 MCP 工具 `{self._remote_name}` 出错: {exc}"


def load_mcp_tools(server_params: "StdioServerParameters" = None) -> List[Tool]:
    if not _MCP_AVAILABLE:
        raise RuntimeError("请先 `pip install mcp` 才能使用 MCP 功能。")
    server_params = server_params or get_default_mcp_server_params()
    infos = asyncio.run(_list_tools_async(server_params))
    return [MCPToolProxy(info, server_params) for info in infos]

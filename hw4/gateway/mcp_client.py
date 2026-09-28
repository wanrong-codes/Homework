import itertools
import json
import logging
from typing import Any, Optional

import httpx

from .tools import Tool, ToolRegistry

log = logging.getLogger("gateway.mcp")


class MCPError(Exception):
    pass


class MCPClient:
    def __init__(self, base_url: str, client: Optional[httpx.AsyncClient] = None):
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=3.0))
        self._ids = itertools.count(1)

    async def _rpc(self, method: str, params: Optional[dict] = None) -> dict:
        payload = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params or {}}
        try:
            r = await self.client.post(f"{self.base_url}/mcp", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise MCPError(f"MCP server unreachable ({self.base_url}): {e!r}") from e
        body = r.json()
        if "error" in body:
            raise MCPError(f"{body['error'].get('code')}: {body['error'].get('message')}")
        return body["result"]

    async def initialize(self) -> dict:
        return await self._rpc("initialize", {"protocolVersion": "2025-03-26",
                                              "clientInfo": {"name": "hw4-gateway", "version": "0.1.0"}})

    async def list_tools(self) -> list[dict]:
        return (await self._rpc("tools/list"))["tools"]

    async def call_tool(self, name: str, arguments: dict) -> Any:
        result = await self._rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            text = "".join(c.get("text", "") for c in result.get("content", []))
            raise MCPError(text or "tool execution failed")
        if "structuredContent" in result:
            return result["structuredContent"]
        return "".join(c.get("text", "") for c in result.get("content", []))

    async def aclose(self):
        await self.client.aclose()


async def sync_remote_tools(registry: ToolRegistry, mcp: MCPClient) -> list[str]:
    remote = await mcp.list_tools()
    registry.unregister_source("mcp")
    registered = []
    for t in remote:
        existing = registry.get(t["name"])
        if existing and existing.source == "local":
            log.warning("skip remote tool %s: name conflicts with a local tool", t["name"])
            continue

        def make_caller(tool_name: str):
            async def call(**kwargs):
                return await mcp.call_tool(tool_name, kwargs)
            return call

        registry.register(Tool(name=t["name"], description=t.get("description", ""),
                               parameters=t["inputSchema"], func=make_caller(t["name"]), source="mcp"))
        registered.append(t["name"])
    return registered

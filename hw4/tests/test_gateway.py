import json

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway import main
from mcp_server.server import app as mcp_app


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        # 把 MCP 客户端指到进程内的 mcp_server（走 ASGI，不需要真的起 9000 端口）
        main.mcp.base_url = "http://mcp"
        main.mcp.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=mcp_app), base_url="http://mcp")
        assert c.post("/mcp/refresh").status_code == 200
        yield c


def sse_events(text: str):
    events = []
    for block in text.strip().split("\n\n"):
        ev, data = "message", None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        events.append((ev, data))
    return events


# 1. 路由 + 模型选择
def test_model_selection(client):
    simple = client.post("/router/select", json={"messages": [{"role": "user", "content": "你好"}]}).json()
    hard = client.post("/router/select", json={"messages": [{"role": "user", "content": "请分析这段代码的复杂度"}]}).json()
    assert simple["tier"] == "cheap" and hard["tier"] == "strong"
    assert simple["provider"] != hard["provider"]  # 两个不同 provider
    forced = client.post("/router/select", json={"messages": [{"role": "user", "content": "hi"}],
                                                 "model": "openai/gpt-4o-mini"}).json()
    assert forced["provider"] == "openai" and forced["model"] == "gpt-4o-mini"


# 2. SSE 流式
def test_streaming(client):
    r = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "流式测试"}], "stream": True})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    assert events[0][0] == "meta" and events[-1][1] == "[DONE]"
    deltas = [json.loads(d)["delta"] for ev, d in events if ev == "message" and d != "[DONE]"]
    assert len(deltas) > 1 and "流式测试" in "".join(deltas)  # 多个增量分片


# 3. RAG
def test_rag(client):
    doc = "SSE 是 Server-Sent Events，用于服务端向客户端单向推送数据。\n\n向量数据库存储 embedding 并做相似度检索。"
    assert client.post("/rag/ingest", json={"doc_id": "notes", "text": doc}).json()["chunks"] >= 1
    hits = client.get("/rag/search", params={"q": "什么是 SSE"}).json()["hits"]
    assert hits and hits[0]["doc_id"] == "notes"
    ans = client.post("/rag/query", json={"question": "什么是 SSE"}).json()
    assert ans["sources"] and "参考资料" in ans["answer"]


# 4. Prompt 管理
def test_prompts(client):
    assert client.post("/prompts/demo/versions", json={"version": "v1", "template": "A:$x"}).status_code == 201
    assert client.post("/prompts/demo/versions", json={"version": "v1", "template": "dup"}).status_code == 409
    client.post("/prompts/demo/versions", json={"version": "v2", "template": "B:$x"})
    assert client.post("/prompts/demo/render", json={"variables": {"x": "1"}}).json()["rendered"] == "A:1"
    client.put("/prompts/demo/active", json={"version": "v2"})
    assert client.post("/prompts/demo/render", json={"variables": {"x": "1"}}).json()["rendered"] == "B:1"
    client.put("/prompts/demo/experiment", json={"enabled": True, "weights": {"v1": 50, "v2": 50}})
    seen = {client.post("/prompts/demo/render", json={"user_id": f"u{i}"}).json()["version"] for i in range(40)}
    assert seen == {"v1", "v2"}
    a = {client.post("/prompts/demo/render", json={"user_id": "alice"}).json()["version"] for _ in range(10)}
    assert len(a) == 1  # 同一用户稳定命中同一版本


# 5. Tool Calling
def test_tools(client):
    names = {t["name"] for t in client.get("/tools").json()["tools"]}
    assert {"calculator", "search_knowledge_base", "get_weather", "get_time", "text_stats"} <= names
    ok = client.post("/tools/calculator/call", json={"arguments": {"expression": "23*7"}}).json()
    assert ok["ok"] and ok["result"]["value"] == 161
    bad = client.post("/tools/calculator/call", json={"arguments": {}}).json()
    assert not bad["ok"]
    r = client.post("/v1/tools/chat", json={"messages": [{"role": "user", "content": "帮我算 12*12"}]}).json()
    assert r["tool_calls"][0]["name"] == "calculator" and r["tool_calls"][0]["result"]["ok"]


# 6. MCP：远程工具可被网关动态调用
def test_mcp(client):
    r = client.post("/tools/get_weather/call", json={"arguments": {"city": "北京"}}).json()
    assert r["ok"] and r["result"]["city"] == "北京"
    assert client.post("/mcp/refresh").json()["registered"]


# 7. Agent：至少 2 步
def test_agent(client):
    r = client.post("/agent/run", json={"question": "北京天气怎么样？另外帮我算 23*7"}).json()
    actions = [s["action"] for s in r["steps"] if s["event"] == "action"]
    assert actions == ["get_weather", "calculator"]
    assert r["steps"][-1]["event"] == "final"

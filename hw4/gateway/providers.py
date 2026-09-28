import asyncio
import json
import re
import uuid
from typing import AsyncIterator, Optional, Protocol

import httpx

from .config import MOCK_LLM, PROVIDERS, ProviderConfig


class ProviderError(Exception):
    """上游 provider 调用失败（网络错误 / 4xx / 5xx），router 会据此做 fallback。"""


class Provider(Protocol):
    name: str

    @property
    def available(self) -> bool: ...

    async def chat(self, model: str, messages: list[dict], *, tools=None, temperature: float = 0.7,
                   max_tokens: Optional[int] = None, stop: Optional[list[str]] = None) -> dict: ...

    def stream(self, model: str, messages: list[dict], *, temperature: float = 0.7,
               max_tokens: Optional[int] = None, stop: Optional[list[str]] = None) -> AsyncIterator[str]: ...

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...


# --------------------------------------------------------------------------- #
# 真实 provider
# --------------------------------------------------------------------------- #
class OpenAICompatProvider:
    def __init__(self, cfg: ProviderConfig, client: Optional[httpx.AsyncClient] = None):
        self.cfg = cfg
        self.name = cfg.name
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    @property
    def available(self) -> bool:
        return bool(self.cfg.api_key)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"}

    def _payload(self, model, messages, temperature, max_tokens, stop, tools=None, stream=False) -> dict:
        payload = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True
        return payload

    async def chat(self, model, messages, *, tools=None, temperature=0.7, max_tokens=None, stop=None) -> dict:
        """非流式调用，返回归一化结果：{content, tool_calls, usage}。"""
        payload = self._payload(model, messages, temperature, max_tokens, stop, tools=tools)
        try:
            r = await self.client.post(f"{self.cfg.base_url}/chat/completions",
                                       headers=self._headers(), json=payload)
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name} network error: {e!r}") from e
        if r.status_code >= 400:
            raise ProviderError(f"{self.name} HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        msg = data["choices"][0]["message"]
        return {
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "usage": data.get("usage", {}),
        }

    async def stream(self, model, messages, *, temperature=0.7, max_tokens=None, stop=None) -> AsyncIterator[str]:
        payload = self._payload(model, messages, temperature, max_tokens, stop, stream=True)
        try:
            async with self.client.stream("POST", f"{self.cfg.base_url}/chat/completions",
                                          headers=self._headers(), json=payload) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode(errors="ignore")
                    raise ProviderError(f"{self.name} HTTP {r.status_code}: {body[:300]}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        yield delta
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name} stream error: {e!r}") from e

    async def embed(self, model, texts) -> list[list[float]]:
        try:
            r = await self.client.post(f"{self.cfg.base_url}/embeddings", headers=self._headers(),
                                       json={"model": model, "input": texts})
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name} network error: {e!r}") from e
        if r.status_code >= 400:
            raise ProviderError(f"{self.name} HTTP {r.status_code}: {r.text[:300]}")
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    async def aclose(self):
        await self.client.aclose()


# --------------------------------------------------------------------------- #
# Mock provider（确定性、无网络）
# --------------------------------------------------------------------------- #
_MATH_RE = re.compile(r"(\d+(?:\.\d+)?(?:\s*[-+*/×÷xX^]\s*\d+(?:\.\d+)?)+)")
_CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "Beijing", "Shanghai", "Tokyo"]


class MockProvider:
    """按规则“假装”是 LLM：支持普通回复、function calling、ReAct 文本格式、RAG 上下文回显。"""

    def __init__(self, name: str = "mock"):
        self.name = name

    @property
    def available(self) -> bool:
        return True

    # ---- 规划：从用户问题里“猜”要调用哪些工具（顺序 = 调用顺序） ----
    @staticmethod
    def _plan(text: str) -> list[tuple[str, dict]]:
        plan: list[tuple[str, dict]] = []
        if "天气" in text or "weather" in text.lower():
            city = next((c for c in _CITIES if c in text), "北京")
            plan.append(("get_weather", {"city": city}))
        m = _MATH_RE.search(text)
        if m:
            expr = m.group(1).replace("×", "*").replace("÷", "/").replace("^", "**")
            expr = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expr)
            plan.append(("calculator", {"expression": expr}))
        if "知识库" in text or "文档" in text:
            plan.append(("search_knowledge_base", {"query": text}))
        return plan

    def _react(self, messages: list[dict]) -> str:
        first_user = next(m["content"] for m in messages if m["role"] == "user")
        question = first_user.split("Question:", 1)[-1].strip()
        observations = [m["content"] for m in messages
                        if m["role"] == "user" and m["content"].startswith("Observation:")]
        plan = self._plan(question)
        if len(observations) < len(plan):
            tool, args = plan[len(observations)]
            return (f"Thought: 我需要调用 {tool} 来获取所需信息。\n"
                    f"Action: {tool}\n"
                    f"Action Input: {json.dumps(args, ensure_ascii=False)}")
        summary = "；".join(o[len("Observation:"):].strip()[:200] for o in observations) or "无需工具即可回答"
        return f"Thought: 我已经掌握足够的信息。\nFinal Answer: （mock）综合工具结果：{summary}"

    def _respond(self, model: str, messages: list[dict], tools) -> tuple[str, list[dict]]:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        if "Action Input" in system:  # ReAct 提示词
            return self._react(messages), []

        last = messages[-1]
        if tools:
            names = {t["function"]["name"] for t in tools}
            if last["role"] == "tool":
                tool_msgs = [m["content"] for m in messages[::-1][: 5] if m["role"] == "tool"]
                return "（mock）根据工具返回的结果：" + " | ".join(reversed(tool_msgs)), []
            calls = [(n, a) for n, a in self._plan(last["content"] or "") if n in names]
            if calls:
                return "", [{
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)},
                } for n, a in calls]

        content = last.get("content") or ""
        ctx = re.search(r"<context>\s*(.*?)\s*</context>", content, re.S)
        if ctx:
            return f"（mock:{model}）根据参考资料回答：{ctx.group(1)[:160]}", []
        return f"（mock:{model}）你说的是：{content[:200]}", []

    async def chat(self, model, messages, *, tools=None, temperature=0.7, max_tokens=None, stop=None) -> dict:
        content, tool_calls = self._respond(model, messages, tools)
        return {"content": content, "tool_calls": tool_calls,
                "usage": {"prompt_tokens": 0, "completion_tokens": len(content), "total_tokens": len(content)}}

    async def stream(self, model, messages, *, temperature=0.7, max_tokens=None, stop=None) -> AsyncIterator[str]:
        content, _ = self._respond(model, messages, None)
        for i in range(0, len(content), 3):  # 每 3 个字符一个“token”，模拟实时输出
            await asyncio.sleep(0.03)
            yield content[i:i + 3]

    async def embed(self, model, texts) -> list[list[float]]:
        raise ProviderError("mock provider has no embeddings; use EMBED_MODE=hash")


def build_providers() -> dict[str, Provider]:
    if MOCK_LLM:
        return {name: MockProvider(name) for name in PROVIDERS}
    return {name: OpenAICompatProvider(cfg) for name, cfg in PROVIDERS.items()}

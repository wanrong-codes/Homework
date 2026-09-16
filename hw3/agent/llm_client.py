from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)


class LLMClient:

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> LLMResponse:
        raise NotImplementedError

    def complete(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        raise NotImplementedError


_MATH_RE = re.compile(
    r"\(?\s*-?\d+(?:\.\d+)?\s*\)?(?:\s*[-+*/%^]\s*\(?\s*-?\d+(?:\.\d+)?\s*\)?){1,}"
)


class MockLLMClient(LLMClient):

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> LLMResponse:
        last_user = self._last_user_message(messages)
        if messages and messages[-1].get("role") == "tool":
            observation = messages[-1]["content"]
            return LLMResponse(content=f"根据工具的结果，答案是：{observation}")

        if not tools:
            return LLMResponse(content=self._direct_answer(last_user))

        tool_names = {t["function"]["name"] for t in tools}
        expr_match = _MATH_RE.search(last_user)

        if "calculator" in tool_names and expr_match:
            expr = expr_match.group().strip()
            return LLMResponse(
                tool_calls=[ToolCall(id="call_1", name="calculator", arguments={"expression": expr})]
            )
        if "weather" in tool_names and any(k in last_user for k in ["天气", "weather"]):
            city = self._extract_city(last_user)
            return LLMResponse(
                tool_calls=[ToolCall(id="call_1", name="weather", arguments={"city": city})]
            )
        if "search" in tool_names and any(
            k in last_user for k in ["搜索", "查一下", "是谁", "是什么", "search", "who", "what", "查询"]
        ):
            return LLMResponse(
                tool_calls=[ToolCall(id="call_1", name="search", arguments={"query": last_user})]
            )
        return LLMResponse(content=self._direct_answer(last_user))

   
    def complete(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        observations = re.findall(r"Observation:\s*(.*)", prompt)
        question_match = re.search(r"Question:\s*(.*)", prompt)
        question = question_match.group(1).strip() if question_match else ""

        needed_actions = self._plan_actions(question)

        if len(observations) >= len(needed_actions):
            summary = "；".join(o.strip() for o in observations) or "已完成"
            return (
                "Thought: 我已经拿到了所有需要的信息，可以直接回答用户了。\n"
                f"Final Answer: {summary}"
            )

        tool_name, thought, args = needed_actions[len(observations)]
        return (
            f"Thought: {thought}\n"
            f"Action: {tool_name}\n"
            f"Action Input: {json.dumps(args, ensure_ascii=False)}"
        )

    def _plan_actions(self, question: str):
        actions = []
        if any(k in question for k in ["天气", "weather"]):
            city = self._extract_city(question)
            actions.append(("weather", "用户想知道天气，我需要用 weather 工具查询。", {"city": city}))

        expr_match = _MATH_RE.search(question)
        if expr_match:
            expr = expr_match.group().strip()
            actions.append(
                ("calculator", "这是一个算术/预算计算问题，我需要用 calculator 工具计算。", {"expression": expr})
            )

        if not actions:
            actions.append(("search", "我需要检索相关信息才能回答这个问题。", {"query": question}))
        return actions

  
    @staticmethod
    def _last_user_message(messages: List[dict]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    @staticmethod
    def _extract_city(text: str) -> str:
        for city in ["tokyo", "paris", "beijing", "new york", "东京", "巴黎", "北京", "纽约"]:
            if city in text.lower():
                mapping = {"东京": "Tokyo", "巴黎": "Paris", "北京": "Beijing", "纽约": "New York"}
                return mapping.get(city, city.title())
        return "Tokyo"

    @staticmethod
    def _direct_answer(text: str) -> str:
        return f"（模拟直接回答，无需调用工具）关于「{text}」，我暂时给出一个通用回复。"



class OpenAICompatLLMClient(LLMClient):


    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "请先 `pip install openai` 才能使用 OpenAICompatLLMClient"
            ) from exc

        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> LLMResponse:
        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": 1024}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        kwargs["temperature"] = 0
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        tool_calls = []
        for tc in getattr(choice, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return LLMResponse(content=choice.content, tool_calls=tool_calls)

    def complete(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stop": stop,
            "max_tokens": 1024,
        }
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


def get_llm_client(provider: Optional[str] = None, **kwargs: Any) -> LLMClient:
    provider = (provider or os.environ.get("LLM_PROVIDER", "mock")).lower()
    if provider == "mock":
        return MockLLMClient()
    if provider in ("openai", "ollama"):
        return OpenAICompatLLMClient(**kwargs)
    raise ValueError(f"未知的 LLM_PROVIDER: {provider}")

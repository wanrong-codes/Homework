import logging
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, Optional

from .config import TIERS
from .providers import Provider, ProviderError

log = logging.getLogger("gateway.router")

COMPLEX_HINTS = (
    "分析", "推理", "证明", "代码", "实现", "设计", "比较", "为什么", "重构", "优化",
    "step by step", "analyze", "prove", "implement", "refactor", "explain why", "design",
)
LONG_CONTEXT_CHARS = 1500


@dataclass
class RouteDecision:
    tier: str       # cheap | strong | explicit
    provider: str
    model: str
    reason: str


class LLMRouter:
    def __init__(self, providers: dict[str, Provider]):
        self.providers = providers

    # ------------------------------------------------------------------ #
    # 模型选择逻辑
    # ------------------------------------------------------------------ #
    def select(self, messages: list[dict], model: str = "auto", has_tools: bool = False) -> RouteDecision:
        if model and "/" in model:
            provider, name = model.split("/", 1)
            if provider not in self.providers:
                raise ValueError(f"unknown provider: {provider}")
            return RouteDecision("explicit", provider, name, "explicitly requested")

        if model in TIERS:
            r = TIERS[model]
            return RouteDecision(model, r.provider, r.model, f"tier '{model}' requested")

        if model not in ("auto", "", None):
            raise ValueError(f"unknown model selector: {model!r} (use auto/cheap/strong/provider/model)")

        text = " ".join((m.get("content") or "") for m in messages)
        last_user = next((m.get("content") or "" for m in reversed(messages) if m["role"] == "user"), "")
        if has_tools:
            tier, reason = "strong", "tool calling / agent needs a stronger model"
        elif len(text) > LONG_CONTEXT_CHARS:
            tier, reason = "strong", f"long context (>{LONG_CONTEXT_CHARS} chars)"
        elif any(h in last_user.lower() for h in COMPLEX_HINTS):
            tier, reason = "strong", "complex-task keywords detected"
        else:
            tier, reason = "cheap", "short/simple query"
        r = TIERS[tier]
        return RouteDecision(tier, r.provider, r.model, reason)

    # ------------------------------------------------------------------ #
    # 候选列表：主路由 + 另一档作为 fallback（要求 provider 已配置 key）
    # ------------------------------------------------------------------ #
    def _candidates(self, d: RouteDecision) -> list[tuple[str, str]]:
        cands = [(d.provider, d.model)]
        if d.tier in TIERS:
            other = TIERS["strong" if d.tier == "cheap" else "cheap"]
            if (other.provider, other.model) not in cands:
                cands.append((other.provider, other.model))
        cands = [c for c in cands if c[0] in self.providers and self.providers[c[0]].available]
        if not cands:
            raise ProviderError("no available provider: set API keys (OPENAI_API_KEY / DEEPSEEK_API_KEY) "
                                "or run with MOCK_LLM=1")
        return cands

    @staticmethod
    def _used(d: RouteDecision, provider: str, model: str) -> RouteDecision:
        if (provider, model) == (d.provider, d.model):
            return d
        return replace(d, provider=provider, model=model, reason=d.reason + f" (fallback → {provider}/{model})")

    # ------------------------------------------------------------------ #
    # 非流式
    # ------------------------------------------------------------------ #
    async def chat(self, d: RouteDecision, messages: list[dict], **kw) -> tuple[dict, RouteDecision]:
        last_err: Optional[Exception] = None
        for provider, model in self._candidates(d):
            try:
                result = await self.providers[provider].chat(model, messages, **kw)
                return result, self._used(d, provider, model)
            except ProviderError as e:
                log.warning("provider %s/%s failed: %s", provider, model, e)
                last_err = e
        raise last_err  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    # 流式：先 yield ("route", decision)，再持续 yield ("delta", text)
    # 已经开始输出后失败就不再 fallback（避免内容拼接错乱）
    # ------------------------------------------------------------------ #
    async def stream(self, d: RouteDecision, messages: list[dict], **kw) -> AsyncIterator[tuple[str, Any]]:
        last_err: Optional[Exception] = None
        for provider, model in self._candidates(d):
            started = False
            used = self._used(d, provider, model)
            try:
                async for delta in self.providers[provider].stream(model, messages, **kw):
                    if not started:
                        started = True
                        yield "route", used
                    yield "delta", delta
                if not started:  
                    yield "route", used
                return
            except ProviderError as e:
                if started:
                    raise
                log.warning("provider %s/%s failed before first token: %s", provider, model, e)
                last_err = e
        raise last_err  

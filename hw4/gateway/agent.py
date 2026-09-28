import json
import re
from typing import AsyncIterator, Optional

from .prompts import PromptManager
from .router import LLMRouter
from .tools import ToolRegistry

REACT_STOP = ["\nObservation:", "Observation:"]  # 让模型在给出 Action 后停下，防止它自己“编造” Observation
MAX_OBS_CHARS = 2000


def _parse_json_object(raw: str) -> dict:
    raw = raw.strip().strip("`").strip()
    raw = re.sub(r"^json\s*", "", raw)
    if not raw:
        return {}
    start = raw.find("{")
    if start == -1:
        raise ValueError("Action Input must be a JSON object")
    obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(obj, dict):
        raise ValueError("Action Input must be a JSON object")
    return obj


def parse_react(text: str) -> dict:
    thought_m = re.search(r"Thought:\s*(.*?)(?=\n\s*(?:Action:|Final Answer:)|\Z)", text, re.S)
    thought = thought_m.group(1).strip() if thought_m else ""
    final_m = re.search(r"Final Answer:\s*(.*)", text, re.S)
    act_m = re.search(r"Action:\s*([^\n]+?)\s*\n\s*Action Input:\s*(.*)", text, re.S)

    # 如果 Action 出现在 Final Answer 之前，以 Action 为准
    if act_m and (not final_m or act_m.start() < final_m.start()):
        try:
            args = _parse_json_object(act_m.group(2))
        except (ValueError, json.JSONDecodeError) as e:
            return {"type": "invalid", "error": f"Action Input 解析失败：{e}"}
        return {"type": "action", "thought": thought, "action": act_m.group(1).strip(), "input": args}
    if final_m:
        return {"type": "final", "thought": thought, "answer": final_m.group(1).strip()}
    return {"type": "invalid", "error": "没有找到 Action 或 Final Answer"}


def _describe_tools(registry: ToolRegistry, names: list[str]) -> str:
    lines = []
    for n in names:
        t = registry.get(n)
        lines.append(f"- {t.name}: {t.description}\n  参数 JSON Schema: "
                     f"{json.dumps(t.parameters, ensure_ascii=False)}")
    return "\n".join(lines)


async def run_agent(question: str, *, router: LLMRouter, registry: ToolRegistry, prompts: PromptManager,
                    tools: Optional[list[str]] = None, model: str = "auto", max_steps: int = 5,
                    user_key: Optional[str] = None) -> AsyncIterator[dict]:
    names = tools or registry.names()
    unknown = [n for n in names if registry.get(n) is None]
    if unknown:
        raise ValueError(f"unknown tools: {unknown}")

    prompt = prompts.resolve("react_agent", user_key=user_key)
    system = prompt.render(tools=_describe_tools(registry, names), tool_names=", ".join(names))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": f"Question: {question}"}]
    decision = router.select(messages, model=model, has_tools=True)
    yield {"event": "start", "question": question, "tools": names,
           "prompt": {"name": prompt.name, "version": prompt.version}, "max_steps": max_steps}

    for step in range(1, max_steps + 1):
        result, used = await router.chat(decision, messages, temperature=0, stop=REACT_STOP)
        text = result["content"].strip()
        parsed = parse_react(text)

        if parsed["type"] == "final":
            yield {"event": "final", "step": step, "thought": parsed["thought"], "answer": parsed["answer"],
                   "route": {"provider": used.provider, "model": used.model}}
            return

        messages.append({"role": "assistant", "content": text})

        if parsed["type"] == "invalid":  # 格式错误：把错误当作 Observation 反馈给模型，让它自我纠正
            obs = {"ok": False, "error": parsed["error"] + "。请严格按 Thought/Action/Action Input 或 Final Answer 格式输出。"}
            yield {"event": "observation", "step": step, "observation": obs}
            messages.append({"role": "user", "content": "Observation: " + json.dumps(obs, ensure_ascii=False)})
            continue

        yield {"event": "thought", "step": step, "thought": parsed["thought"]}
        yield {"event": "action", "step": step, "action": parsed["action"], "input": parsed["input"]}

        if parsed["action"] not in names:
            obs = {"tool": parsed["action"], "ok": False, "error": f"tool not allowed. available: {names}"}
        else:
            obs = await registry.execute(parsed["action"], parsed["input"])
        yield {"event": "observation", "step": step, "observation": obs}
        messages.append({"role": "user",
                         "content": "Observation: " + json.dumps(obs, ensure_ascii=False)[:MAX_OBS_CHARS]})

    yield {"event": "final", "step": max_steps, "truncated": True,
           "answer": f"（已达到最大步数 {max_steps}，仍未得出最终答案）"}

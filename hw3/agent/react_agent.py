from __future__ import annotations

import json
import re
from typing import List, Optional

from .llm_client import LLMClient
from .schemas import to_react_tool_descriptions
from .tools import Tool

_REACT_SYSTEM_TEMPLATE = """你是一个可以使用工具来回答问题的智能体。请严格按照以下格式，一步一步地思考并行动：

Thought: 你的推理过程
Action: 要调用的工具名（必须是下面工具列表中的一个）
Action Input: 一个 JSON 对象，表示传给工具的参数

在你收到 Observation（工具返回的结果）之后，你可以继续 Thought/Action/Action Input，
直到你确信可以回答问题为止，这时请输出：

重要：每次只能输出一个 Thought + 一个 Action + 一个 Action Input，然后停止，
等待真实的 Observation 返回后再继续。禁止自己假设或编造 Observation 的内容，
也禁止一次性输出多个 Action。

重要：Final Answer 必须完整回答用户问题里提到的每一个子问题，
把之前所有轮次的 Observation 结果都综合进去，不能只回答最后一步的结果。

Thought: 我现在可以回答了
Final Answer: 最终答案

可用工具：
{tool_descriptions}

Question: {question}
{scratchpad}"""

_ACTION_RE = re.compile(r"Action:\s*(\w+)\s*", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(
    r"Action Input:\s*```(?:json)?\s*(\{.*?\})\s*```|Action Input:\s*(\{.*\})",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)


class ReActAgent:
    def __init__(self, llm_client: LLMClient, tools: List[Tool]):
        self.llm_client = llm_client
        self.tools_by_name = {t.name: t for t in tools}
        self.tool_descriptions = to_react_tool_descriptions(tools)

    def run(self, question: str, max_steps: int = 6, verbose: bool = True) -> str:
        scratchpad = ""
        for step in range(1, max_steps + 1):
            prompt = _REACT_SYSTEM_TEMPLATE.format(
                tool_descriptions=self.tool_descriptions,
                question=question,
                scratchpad=scratchpad,
            )
            completion = self.llm_client.complete(
                prompt, stop=["Observation:", "\nThought:", "\nQuestion:"]
            )
            if verbose:
                print(f"  [step {step}] {completion.strip()}")

            final_match = _FINAL_ANSWER_RE.search(completion)
            action_match_check = _ACTION_RE.search(completion)
            if final_match and (not action_match_check or final_match.start() < action_match_check.start()):
                return final_match.group(1).strip()

            action_match = _ACTION_RE.search(completion)
            input_match = _ACTION_INPUT_RE.search(completion)
            if not action_match:
                return completion.strip()

            tool_name = action_match.group(1).strip()
            try:
                raw_json = (input_match.group(1) or input_match.group(2)) if input_match else None
                args = json.loads(raw_json) if raw_json else {}
            except json.JSONDecodeError:
                args = {}

            tool = self.tools_by_name.get(tool_name)
            if tool is None:
                observation = f"错误：未找到名为 {tool_name} 的工具"
            else:
                try:
                    observation = tool.run(**args)
                except TypeError as exc:
                    observation = f"错误：调用 {tool_name} 时参数不对（{exc}），请检查 Action Input 是否包含了所有必填参数"
                except Exception as exc:  # noqa: BLE001
                    observation = f"错误：执行 {tool_name} 时出错：{exc}"
            if verbose:
                print(f"  [step {step}] Observation: {observation}")

            scratchpad += (
                f"\nThought: {self._extract_thought(completion)}"
                f"\nAction: {tool_name}"
                f"\nAction Input: {json.dumps(args, ensure_ascii=False)}"
                f"\nObservation: {observation}\n"
            )

        return "达到最大步数仍未得到最终答案。"

    @staticmethod
    def _extract_thought(completion: str) -> str:
        match = re.search(r"Thought:\s*(.*?)(?:\nAction:|$)", completion, re.DOTALL)
        return match.group(1).strip() if match else ""

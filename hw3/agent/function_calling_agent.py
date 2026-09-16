from __future__ import annotations
import json
from typing import List, Optional

from .llm_client import LLMClient, LLMResponse
from .schemas import to_function_calling_schemas
from .tools import Tool


class FunctionCallingAgent:
    def __init__(self, llm_client: LLMClient, tools: List[Tool], system_prompt: Optional[str] = None):
        self.llm_client = llm_client
        self.tools_by_name = {t.name: t for t in tools}
        self.tool_schemas = to_function_calling_schemas(tools)
        self.system_prompt = system_prompt or (
            "你是一个可以调用工具的助手。当需要精确计算或查找事实信息时，"
            "请调用合适的工具，而不要凭空猜测答案。"
        )

    def run(self, user_message: str, max_iterations: int = 5, verbose: bool = True) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for step in range(1, max_iterations + 1):
            response: LLMResponse = self.llm_client.chat(messages, tools=self.tool_schemas)

            if response.tool_calls:
                # LLM 决定调用一个或多个工具
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                                },
                            }
                            for tc in response.tool_calls
                        ],
                    }
                )


                for tc in response.tool_calls:
                    tool = self.tools_by_name.get(tc.name)
                    if tool is None:
                        result = f"错误：未找到名为 {tc.name} 的工具"
                    else:
                        result = tool.run(**tc.arguments)
                    if verbose:
                        print(f"  [step {step}] 调用工具 `{tc.name}`({tc.arguments}) -> {result}")
                    messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result})
                continue  # 把工具结果喂回去，让 LLM 继续决策 / 总结

            # 没有工具调用，说明 LLM 给出了最终答案
            return response.content or ""

        return "达到最大迭代次数仍未得到最终答案。"

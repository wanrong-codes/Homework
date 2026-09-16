from __future__ import annotations

from typing import List, Optional

from .function_calling_agent import FunctionCallingAgent
from .llm_client import LLMClient
from .memory import MemoryStore
from .tools import CalculatorTool, SearchTool, Tool, WeatherTool


class Skill:

    name: str
    description: str

    def __init__(self, llm_client: LLMClient, tools: List[Tool], system_prompt: str):
        self.agent = FunctionCallingAgent(llm_client, tools, system_prompt=system_prompt)


class TravelPlanningSkill(Skill):

    name = "travel_planning"
    description = "根据目的地、天数和预算，生成一个简单的旅行计划，会结合天气、目的地信息和预算拆分。"

    def __init__(self, llm_client: LLMClient, memory: Optional[MemoryStore] = None):
        self.memory = memory
        tools = [SearchTool(), WeatherTool(), CalculatorTool()]
        system_prompt = (
            "你是一个旅行规划助手。你可以使用 search 查询目的地相关信息、"
            "使用 weather 查询天气、使用 calculator 做预算拆分计算。"
            "请结合这些信息给出简洁、可执行的旅行建议。"
        )
        super().__init__(llm_client, tools, system_prompt)

    def plan_trip(self, user_id: str, destination: str, days: int, budget: float, verbose: bool = True) -> str:
        preference_text = ""
        if self.memory is not None:
            preference_text = self.memory.format_for_prompt(user_id)

        prompt = (
            f"请帮我规划一个 {days} 天、预算 {budget} 元的 {destination} 旅行计划。"
            f"先查一下 {destination} 的天气和基本信息，再帮我把预算按 住宿/餐饮/交通/其他 做一个大致拆分（用 calculator 算出每项金额）。"
        )
        if preference_text:
            prompt = f"{preference_text}\n\n{prompt}"

        return self.agent.run(prompt, verbose=verbose)


class CustomerSupportSkill(Skill):

    name = "customer_support"
    description = "结合知识库检索和金额计算，回答售后/客服类问题（例如退款、折扣计算）。"

    def __init__(self, llm_client: LLMClient, memory: Optional[MemoryStore] = None):
        self.memory = memory
        tools = [SearchTool(), CalculatorTool()]
        system_prompt = (
            "你是一个客服助手。遇到事实性问题请用 search 工具核实；"
            "遇到金额/折扣计算请用 calculator 工具，不要自己心算。"
        )
        super().__init__(llm_client, tools, system_prompt)

    def answer(self, user_id: str, question: str, verbose: bool = True) -> str:
        preference_text = self.memory.format_for_prompt(user_id) if self.memory else ""
        prompt = f"{preference_text}\n\n{question}" if preference_text else question
        return self.agent.run(prompt, verbose=verbose)

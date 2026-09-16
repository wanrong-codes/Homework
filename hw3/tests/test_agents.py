import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.function_calling_agent import FunctionCallingAgent
from agent.llm_client import MockLLMClient
from agent.memory import MemoryStore
from agent.react_agent import ReActAgent
from agent.skills import TravelPlanningSkill
from agent.tools import CalculatorTool, SearchTool


def test_calculator_tool_direct():
    tool = CalculatorTool()
    assert tool.run(expression="(2+3)*4") == "20"


def test_search_tool_direct():
    tool = SearchTool()
    result = tool.run(query="python")
    assert "Python" in result or "python" in result.lower()


def test_function_calling_agent_uses_calculator():
    llm = MockLLMClient()
    agent = FunctionCallingAgent(llm, tools=[CalculatorTool(), SearchTool()])
    answer = agent.run("请计算 12 * 6", verbose=False)
    assert "72" in answer


def test_react_agent_multi_step():
    llm = MockLLMClient()
    agent = ReActAgent(llm, tools=[CalculatorTool(), SearchTool()])
    answer = agent.run("计算 5 + 7", verbose=False)
    assert "12" in answer


def test_memory_store_roundtrip(tmp_path=None):
    path = "test_memory_store.json"
    if os.path.exists(path):
        os.remove(path)
    mem = MemoryStore(path=path)
    mem.add_preference("u1", "style", "budget")
    assert mem.get_preference("u1", "style") == "budget"
    assert "style" in mem.format_for_prompt("u1")
    os.remove(path)


def test_skill_plan_trip_runs():
    llm = MockLLMClient()
    mem_path = "test_skill_memory.json"
    if os.path.exists(mem_path):
        os.remove(mem_path)
    memory = MemoryStore(path=mem_path)
    memory.add_preference("u1", "style", "budget")
    skill = TravelPlanningSkill(llm, memory=memory)
    result = skill.plan_trip(user_id="u1", destination="Tokyo", days=3, budget=1500, verbose=False)
    assert isinstance(result, str) and len(result) > 0
    os.remove(mem_path)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print("\n所有测试通过 ✅")

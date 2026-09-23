import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm_client import MockLLMClient
from agent.multi_agent_workflow import build_multi_agent_graph, run_multi_agent_task
from agent.tools import CalculatorTool, SearchTool


def test_multi_agent_workflow_runs_end_to_end():
    llm = MockLLMClient()
    tools = [CalculatorTool(), SearchTool()]
    result = run_multi_agent_task("查一下python是什么；然后计算3*4", llm, tools)

    assert "plan" in result and len(result["plan"]) >= 1
    assert "final_answer" in result and len(result["final_answer"]) > 0
    # Mock 模式下 critic 总是 APPROVE，所以每个 plan 步骤都应该走完一次
    assert len(result["step_history"]) == len(result["plan"])


def test_graph_compiles_and_is_reusable():
    llm = MockLLMClient()
    app = build_multi_agent_graph(llm, [CalculatorTool()])
    assert app is not None
    result = app.invoke({"task": "计算 10 + 5", "user_id": "u1"})
    assert "10" in result["final_answer"] or "15" in result["final_answer"] or len(result["final_answer"]) > 0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print("\n所有测试通过 ✅")

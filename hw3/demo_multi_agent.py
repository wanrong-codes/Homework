from agent.llm_client import get_llm_client
from agent.mcp_client import load_mcp_tools
from agent.memory import MemoryStore
from agent.multi_agent_workflow import run_multi_agent_task
from agent.tools import default_toolset


def main():
    llm = get_llm_client()  # 读取 .env 里的 LLM_PROVIDER，默认 mock，离线可跑

    tools = default_toolset()
    try:
        mcp_tools = load_mcp_tools()
        tools = tools + mcp_tools
        print(f"已通过 MCP 加载 {len(mcp_tools)} 个工具：{[t.name for t in mcp_tools]}\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[提示] 未能连接 MCP server（{exc}），本次演示仅使用本地工具。\n")

    memory = MemoryStore(path="demo_memory_store.json")
    memory.add_preference("u1", "style", "偏好性价比高、不要太赶的行程")

    task = (
        "帮我规划一个3天东京旅行，预算6000元；查一下东京天气；"
        "并把最终的行程要点记一条笔记。"
    )

    result = run_multi_agent_task(task, llm, tools, memory=memory, user_id="u1")

    print("=== Plan (Planner Agent) ===")
    for i, step in enumerate(result["plan"], 1):
        print(f"{i}. {step}")

    print("\n=== Step History (Worker + Critic) ===")
    for h in result["step_history"]:
        print(f"- {h['step']}\n  -> {h['result']}\n")

    print("=== Final Answer ===")
    print(result["final_answer"])


if __name__ == "__main__":
    main()

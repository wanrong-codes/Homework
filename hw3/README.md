
```bash
cd hw3
python3 -m venv .venv && source .venv/bin/activate   # 可选
pip install -r requirements.txt                       # Mock 模式其实不需要装 openai，也能跑

python demo_function_calling.py
python demo_react.py
python demo_skills_and_memory.py

python -m pytest tests/ -v      # 或者 python tests/test_agents.py
```

## MCP Server 集成


```bash
python demo_mcp.py   # 直接连接 MCP server，通过 MCP 协议 list_tools / call_tool
```

## Multi-Agent Workflow（LangGraph：Planner / Worker / Critic）

`agent/multi_agent_workflow.py` 用 LangGraph 的 `StateGraph` 搭了一个
Planner -> Worker -> Critic 的循环：

- **Planner**：把用户任务拆成 2-4 个子步骤
- **Worker**：执行当前子步骤，需要时调用工具（本地工具 + MCP 工具都可以），
  会读取 Memory 里的用户偏好并注入 prompt
- **Critic**：审查 Worker 的结果，APPROVE 就推进到下一步，REVISE 就带着
  修改意见打回给 Worker 重做（最多重试 `MAX_REVISIONS` 次，之后接受当前结果）

```bash
python demo_multi_agent.py    # Planner/Worker/Critic + MCP + 本地工具 + Memory 的完整演示
python -m pytest tests/test_multi_agent.py -v   # 离线测试（用 MockLLMClient，不需要真实LLM）
```


## 切换到真实 LLM

复制 `.env.example` 为 `.env`（或者直接 `export`），设置：

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-xxxx
export OPENAI_MODEL=gpt-4o-mini
```


```bash
export LLM_PROVIDER=openai
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export OPENAI_MODEL=llama3.1
```


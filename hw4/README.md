# hw4 — AI Gateway (FastAPI)


## 运行

```bash
pip install -r requirements.txt
cp .env.example .env        # 填 OPENAI_API_KEY / DEEPSEEK_API_KEY；没有 Key 就 export MOCK_LLM=1
export $(grep -v '^#' .env | xargs)   # 或者直接 export 需要的变量

# 终端 1：MCP 工具服务
uvicorn mcp_server.server:app --port 9000
# 终端 2：网关（启动时会自动拉取 MCP 工具）
uvicorn gateway.main:app --reload --port 8000
```
文档：http://localhost:8000/docs

测试：`pytest -q`（自动使用 MockProvider，不需要 Key）

## 示例

```bash
# 1. 路由：看模型选择结果（简单问题 → cheap，复杂/长文本/带工具 → strong）
curl -s localhost:8000/router/select -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"请分析一下这段代码"}]}'

# 2. SSE 流式（-N 关闭 curl 缓冲）
curl -N localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"用三句话介绍 SSE"}],"stream":true}'

# 3. RAG
curl -s localhost:8000/rag/ingest -H 'content-type: application/json' \
  -d '{"doc_id":"notes","text":"SSE 是 Server-Sent Events，用于服务端单向推送流式数据。\n\n向量数据库用于存储 embedding 并做相似度检索。"}'
curl -s localhost:8000/rag/query -H 'content-type: application/json' -d '{"question":"什么是 SSE？"}'
curl -F file=@notes.md -F doc_id=notes2 localhost:8000/rag/ingest_file      # 文件上传

# 4. Prompt：新建版本 → 开 A/B → 按 user_id 稳定分流 → 看统计
curl -s localhost:8000/prompts/chat_assistant/versions -H 'content-type: application/json' \
  -d '{"version":"v3","template":"你是一位严谨的助教，回答时先给结论再给理由。"}'
curl -s -X PUT localhost:8000/prompts/chat_assistant/experiment -H 'content-type: application/json' \
  -d '{"enabled":true,"weights":{"v1":50,"v2":50}}'
curl -s localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"prompt_name":"chat_assistant","user_id":"alice","messages":[{"role":"user","content":"你好"}]}'
curl -s localhost:8000/prompts/chat_assistant/stats

# 5. Tool Calling（LLM 决定调用哪些工具 → 网关执行 → 返回结构化结果）
curl -s localhost:8000/tools
curl -s localhost:8000/v1/tools/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"北京天气怎么样？再算一下 23*7"}]}'

# 6. MCP：远程工具（get_weather / get_time / text_stats）已经动态注册进网关
curl -s localhost:9000/tools                                   # MCP 服务器自己的工具列表
curl -s -X POST localhost:8000/mcp/refresh                     # 服务器改了工具后热更新
curl -s localhost:8000/tools/get_weather/call -H 'content-type: application/json' -d '{"arguments":{"city":"上海"}}'

# 7. Agent（ReAct，stream=true 时每一步以 SSE 事件推送）
curl -s localhost:8000/agent/run -H 'content-type: application/json' \
  -d '{"question":"北京现在天气如何？并计算 23*7"}'
curl -N localhost:8000/agent/run -H 'content-type: application/json' \
  -d '{"question":"北京现在天气如何？并计算 23*7","stream":true}'
```


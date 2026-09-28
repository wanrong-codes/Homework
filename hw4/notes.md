# AI Gateway 学习笔记

SSE 是 Server-Sent Events，一种基于 HTTP 的单向推送技术。服务端保持连接不断开，持续向客户端发送 `data: ...` 格式的文本消息，非常适合大模型的逐 token 流式输出。

FastAPI 通过 StreamingResponse 返回异步生成器，并把 media_type 设为 text/event-stream，就可以实现 SSE。

RAG 是检索增强生成（Retrieval-Augmented Generation）。流程是：先把文档切成小块并转成 embedding 向量存入向量库，提问时检索出最相似的几个片段，再把这些片段作为上下文交给大模型生成回答。

向量数据库用于存储 embedding 并做相似度检索，常用的相似度度量是余弦相似度。切块时通常会保留一部分 overlap，避免一句话被切断后丢失上下文。

Prompt 版本管理的核心思想是：提示词像代码一样有版本号，可以回滚，也可以做 A/B 测试，比较不同版本的效果。

MCP 是 Model Context Protocol，用统一的协议把工具暴露给大模型应用。工具服务器通过 tools/list 告知有哪些工具，通过 tools/call 执行工具。

ReAct 是 Reason 加 Act 的缩写：模型先思考（Thought），再选择工具行动（Action），拿到工具返回的观察结果（Observation）后继续思考，直到给出最终答案。

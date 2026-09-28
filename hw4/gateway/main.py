import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import run_agent
from .config import DATA_DIR, MCP_SERVER_URL, MOCK_LLM, PROMPT_DIR, TIERS
from .mcp_client import MCPClient, MCPError, sync_remote_tools
from .prompts import PromptConflict, PromptError, PromptManager, PromptNotFound
from .providers import ProviderError, build_providers
from .rag import RAGService, VectorStore, build_embedder
from .router import LLMRouter
from .tools import ToolRegistry, build_local_tools

log = logging.getLogger("gateway")
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- #
# 组件装配（模块级单例）
# --------------------------------------------------------------------------- #
providers = build_providers()
router = LLMRouter(providers)
prompts = PromptManager(PROMPT_DIR)
rag = RAGService(build_embedder(providers), VectorStore(DATA_DIR / "store.json"))
registry = ToolRegistry()
for _t in build_local_tools(rag):
    registry.register(_t)
mcp = MCPClient(MCP_SERVER_URL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:  
        names = await sync_remote_tools(registry, mcp)
        log.info("registered remote MCP tools: %s", names)
    except MCPError as e:
        log.warning("MCP server not available at startup (%s). Use POST /mcp/refresh later.", e)
    yield
    await mcp.aclose()
    for p in providers.values():
        if hasattr(p, "aclose"):
            await p.aclose()


app = FastAPI(title="AI Gateway", lifespan=lifespan)


@app.exception_handler(PromptNotFound)
async def _prompt_not_found(_: Request, e: PromptNotFound):
    return JSONResponse(status_code=404, content={"detail": str(e)})


@app.exception_handler(PromptConflict)
async def _prompt_conflict(_: Request, e: PromptConflict):
    return JSONResponse(status_code=409, content={"detail": str(e)})


@app.exception_handler(PromptError)
async def _prompt_error(_: Request, e: PromptError):
    return JSONResponse(status_code=400, content={"detail": str(e)})


# --------------------------------------------------------------------------- #
# SSE 工具
# --------------------------------------------------------------------------- #
SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}


def sse(data, event: Optional[str] = None) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return (f"event: {event}\n" if event else "") + f"data: {payload}\n\n"


def sse_response(gen) -> StreamingResponse:
    return StreamingResponse(gen, media_type="text/event-stream", headers=SSE_HEADERS)


# --------------------------------------------------------------------------- #
# 请求模型
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str = "auto"  # auto | cheap | strong | provider/model
    stream: bool = False
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, gt=0)
    prompt_name: Optional[str] = None            # 使用 Prompt 管理系统里的系统提示词
    prompt_vars: dict[str, str] = Field(default_factory=dict)
    user_id: Optional[str] = None                # A/B 分流的稳定分桶 key


class IngestRequest(BaseModel):
    doc_id: str
    text: str
    chunk_size: int = Field(500, ge=50, le=4000)
    overlap: int = Field(80, ge=0, le=1000)


class RAGQuery(BaseModel):
    question: str
    top_k: int = Field(4, ge=1, le=10)
    model: str = "auto"
    stream: bool = False
    user_id: Optional[str] = None


class NewVersion(BaseModel):
    version: str
    template: str
    activate: bool = False


class ActivateReq(BaseModel):
    version: str


class ExperimentReq(BaseModel):
    enabled: bool = True
    weights: dict[str, float] = Field(default_factory=dict)


class RenderReq(BaseModel):
    variables: dict[str, str] = Field(default_factory=dict)
    version: Optional[str] = None
    user_id: Optional[str] = None


class ToolCallReq(BaseModel):
    arguments: dict = Field(default_factory=dict)


class ToolChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    tools: Optional[list[str]] = None  # 不填 = 使用全部已注册工具
    model: str = "auto"
    max_rounds: int = Field(3, ge=1, le=8)
    temperature: float = Field(0, ge=0, le=2)


class AgentRequest(BaseModel):
    question: str
    tools: Optional[list[str]] = None
    model: str = "auto"
    max_steps: int = Field(5, ge=1, le=10)
    stream: bool = False
    user_id: Optional[str] = None


# --------------------------------------------------------------------------- #
# 基础 & 路由
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    return {"status": "ok", "mock_llm": MOCK_LLM,
            "providers": {n: p.available for n, p in providers.items()},
            "tiers": {k: asdict(v) for k, v in TIERS.items()},
            "tools": registry.names()}


@app.post("/router/select")
async def router_select(req: ChatRequest):
    """只做模型选择、不调用 LLM —— 方便观察路由逻辑。"""
    try:
        d = router.select([m.model_dump() for m in req.messages], req.model)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return asdict(d)


# --------------------------------------------------------------------------- #
# 聊天（含 SSE 流式）
# --------------------------------------------------------------------------- #
def _apply_prompt(req: ChatRequest, messages: list[dict]) -> Optional[dict]:
    if not req.prompt_name:
        return None
    p = prompts.resolve(req.prompt_name, user_key=req.user_id)
    messages.insert(0, {"role": "system", "content": p.render(**req.prompt_vars)})
    return {"name": p.name, "version": p.version, "source": p.source}


async def _sse_llm(decision, messages, *, temperature, max_tokens, prompt_info=None, sources=None):
    """把 router.stream 的输出转成 SSE：
       event: sources（可选，RAG 引用） → event: meta（路由信息） → data: {"delta": "..."} × N → data: [DONE]"""
    try:
        if sources is not None:
            yield sse({"sources": sources}, event="sources")
        async for kind, payload in router.stream(decision, messages, temperature=temperature,
                                                 max_tokens=max_tokens):
            if kind == "route":
                yield sse({"route": asdict(payload), "prompt": prompt_info}, event="meta")
            else:
                yield sse({"delta": payload})
        yield sse("[DONE]")
    except ProviderError as e:
        yield sse({"error": str(e)}, event="error")


@app.post("/v1/chat")
async def chat(req: ChatRequest):
    messages = [m.model_dump() for m in req.messages]
    prompt_info = _apply_prompt(req, messages)
    try:
        decision = router.select(messages, req.model)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if req.stream:
        return sse_response(_sse_llm(decision, messages, temperature=req.temperature,
                                     max_tokens=req.max_tokens, prompt_info=prompt_info))
    try:
        result, used = await router.chat(decision, messages, temperature=req.temperature,
                                         max_tokens=req.max_tokens)
    except ProviderError as e:
        raise HTTPException(502, str(e))
    return {"content": result["content"], "usage": result["usage"], "route": asdict(used), "prompt": prompt_info}


# --------------------------------------------------------------------------- #
# RAG
# --------------------------------------------------------------------------- #
@app.post("/rag/ingest")
async def rag_ingest(req: IngestRequest):
    try:
        return await rag.ingest(req.doc_id, req.text, req.chunk_size, req.overlap)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except ProviderError as e:
        raise HTTPException(502, str(e))


@app.post("/rag/ingest_file")
async def rag_ingest_file(file: UploadFile = File(...), doc_id: Optional[str] = Form(None),
                          chunk_size: int = Form(500), overlap: int = Form(80)):
    text = (await file.read()).decode("utf-8", errors="replace")
    try:
        return await rag.ingest(doc_id or file.filename or "upload", text, chunk_size, overlap)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except ProviderError as e:
        raise HTTPException(502, str(e))


@app.get("/rag/search")
async def rag_search(q: str, top_k: int = 4):
    return {"query": q, "hits": await rag.retrieve(q, top_k)}


@app.get("/rag/docs")
async def rag_docs():
    return rag.store.stats()


@app.delete("/rag/docs/{doc_id}")
async def rag_delete(doc_id: str):
    return {"doc_id": doc_id, "removed_chunks": rag.store.delete_doc(doc_id)}


@app.post("/rag/query")
async def rag_query(req: RAGQuery):
    hits = await rag.retrieve(req.question, req.top_k)
    p = prompts.resolve("rag_answer", user_key=req.user_id)
    prompt_text = p.render(context=rag.build_context(hits), question=req.question)
    messages = [{"role": "user", "content": prompt_text}]
    prompt_info = {"name": p.name, "version": p.version, "source": p.source}
    sources = [{k: h[k] for k in ("doc_id", "chunk_id", "score")} | {"snippet": h["text"][:120]} for h in hits]
    try:
        decision = router.select(messages, req.model)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if req.stream:
        return sse_response(_sse_llm(decision, messages, temperature=0.2, max_tokens=None,
                                     prompt_info=prompt_info, sources=sources))
    try:
        result, used = await router.chat(decision, messages, temperature=0.2)
    except ProviderError as e:
        raise HTTPException(502, str(e))
    return {"answer": result["content"], "sources": sources, "route": asdict(used), "prompt": prompt_info}


# --------------------------------------------------------------------------- #
# Prompt 管理
# --------------------------------------------------------------------------- #
@app.get("/prompts")
async def prompts_list():
    return {"prompts": [prompts.info(n) for n in prompts.names()]}


@app.get("/prompts/{name}")
async def prompts_get(name: str):
    info = prompts.info(name)
    info["templates"] = {v: prompts._template(name, v) for v in info["versions"]}
    return info


@app.post("/prompts/{name}/versions", status_code=201)
async def prompts_create_version(name: str, req: NewVersion):
    return prompts.create_version(name, req.version, req.template, req.activate)


@app.put("/prompts/{name}/active")
async def prompts_activate(name: str, req: ActivateReq):
    return prompts.set_active(name, req.version)


@app.put("/prompts/{name}/experiment")
async def prompts_experiment(name: str, req: ExperimentReq):
    """开启/关闭 A/B 实验，例如 {"enabled": true, "weights": {"v1": 70, "v2": 30}}"""
    return prompts.set_experiment(name, req.enabled, req.weights)


@app.get("/prompts/{name}/stats")
async def prompts_stats(name: str):
    prompts.versions(name)  # 校验存在
    return prompts.get_stats(name)


@app.post("/prompts/{name}/render")
async def prompts_render(name: str, req: RenderReq):
    p = prompts.resolve(name, version=req.version, user_key=req.user_id)
    return {"name": p.name, "version": p.version, "source": p.source, "rendered": p.render(**req.variables)}


# --------------------------------------------------------------------------- #
# Tools / MCP
# --------------------------------------------------------------------------- #
@app.get("/tools")
async def tools_list():
    return {"tools": registry.describe_all()}


@app.post("/tools/{name}/call")
async def tools_call(name: str, req: ToolCallReq):
    return await registry.execute(name, req.arguments)


@app.post("/mcp/refresh")
async def mcp_refresh():
    try:
        names = await sync_remote_tools(registry, mcp)
    except MCPError as e:
        raise HTTPException(502, str(e))
    return {"server": MCP_SERVER_URL, "registered": names, "all_tools": registry.names()}


@app.post("/v1/tools/chat")
async def tools_chat(req: ToolChatRequest):
    names = req.tools or registry.names()
    unknown = [n for n in names if registry.get(n) is None]
    if unknown:
        raise HTTPException(400, f"unknown tools: {unknown}")
    schemas = registry.openai_schemas(names)
    messages: list[dict] = [m.model_dump() for m in req.messages]
    try:
        decision = router.select(messages, req.model, has_tools=True)
    except ValueError as e:
        raise HTTPException(400, str(e))

    trace: list[dict] = []

    async def run_call(tc: dict) -> dict:
        fn = tc["function"]
        if fn["name"] not in names:
            result = {"tool": fn["name"], "ok": False, "error": "tool not allowed for this request"}
        else:
            result = await registry.execute(fn["name"], fn.get("arguments"))
        return {"id": tc["id"], "name": fn["name"], "arguments": fn.get("arguments"), "result": result}

    try:
        for round_no in range(1, req.max_rounds + 1):
            res, used = await router.chat(decision, messages, tools=schemas, temperature=req.temperature)
            if not res["tool_calls"]:
                return {"content": res["content"], "tool_calls": trace, "rounds": round_no,
                        "route": asdict(used)}
            messages.append({"role": "assistant", "content": res["content"] or None,
                             "tool_calls": res["tool_calls"]})
            executed = await asyncio.gather(*(run_call(tc) for tc in res["tool_calls"]))  # 同一轮的多个调用并行
            for item in executed:
                trace.append(item)
                messages.append({"role": "tool", "tool_call_id": item["id"],
                                 "content": json.dumps(item["result"], ensure_ascii=False)})
        # 轮数用尽：不再给工具，强制模型收尾
        res, used = await router.chat(decision, messages, temperature=req.temperature)
        return {"content": res["content"], "tool_calls": trace, "rounds": req.max_rounds,
                "route": asdict(used), "truncated": True}
    except ProviderError as e:
        raise HTTPException(502, str(e))


# --------------------------------------------------------------------------- #
# Agent（ReAct）
# --------------------------------------------------------------------------- #
@app.post("/agent/run")
async def agent_run(req: AgentRequest):
    gen = run_agent(req.question, router=router, registry=registry, prompts=prompts, tools=req.tools,
                    model=req.model, max_steps=req.max_steps, user_key=req.user_id)

    if req.stream:
        async def events():
            try:
                async for ev in gen:
                    yield sse(ev, event=ev["event"])
                yield sse("[DONE]")
            except (ProviderError, ValueError) as e:
                yield sse({"error": str(e)}, event="error")
        return sse_response(events())

    steps: list[dict] = []
    try:
        async for ev in gen:
            steps.append(ev)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except ProviderError as e:
        raise HTTPException(502, str(e))
    final = steps[-1]
    return {"answer": final.get("answer"), "truncated": final.get("truncated", False),
            "steps": [s for s in steps if s["event"] != "start"], "tools": steps[0]["tools"]}

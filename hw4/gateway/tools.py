import ast
import inspect
import json
import operator
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jsonschema


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          
    func: Callable[..., Any]  
    source: str = "local"     

    def openai_schema(self) -> dict:
        """OpenAI function-calling 格式。"""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def describe(self) -> dict:
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters, "source": self.source}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister_source(self, source: str) -> None:
        self._tools = {n: t for n, t in self._tools.items() if t.source != source}

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def describe_all(self) -> list[dict]:
        return [t.describe() for t in self._tools.values()]

    def openai_schemas(self, names: Optional[list[str]] = None) -> list[dict]:
        return [t.openai_schema() for n, t in self._tools.items() if names is None or n in names]

    async def execute(self, name: str, arguments: Any) -> dict:
        t0 = time.perf_counter()

        def done(**kw):
            return {"tool": name, **kw, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2)}

        tool = self._tools.get(name)
        if tool is None:
            return done(ok=False, error=f"unknown tool: {name}")
        try:
            # LLM 给的 arguments 通常是 JSON 字符串
            if isinstance(arguments, str):
                args = json.loads(arguments) if arguments.strip() else {}
            else:
                args = arguments or {}
            jsonschema.validate(args, tool.parameters)  # 参数校验
            res = tool.func(**args)
            if inspect.isawaitable(res):
                res = await res
            return done(ok=True, result=res)
        except json.JSONDecodeError as e:
            return done(ok=False, error=f"arguments is not valid JSON: {e}")
        except jsonschema.ValidationError as e:
            return done(ok=False, error=f"invalid arguments: {e.message}")
        except Exception as e:  # 工具内部错误也以结构化形式返回给 LLM，让它自己修正
            return done(ok=False, error=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# 工具 1：calculator（用 ast 白名单求值，绝不使用 eval）
# --------------------------------------------------------------------------- #
_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expression: str) -> float:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression (only numbers and + - * / // % ** are allowed)")

    return ev(ast.parse(expression.strip(), mode="eval"))


def calculator(expression: str) -> dict:
    return {"expression": expression, "value": safe_eval(expression)}


CALCULATOR_SCHEMA = {
    "type": "object",
    "properties": {"expression": {"type": "string", "description": "算术表达式，例如 '23 * 7 + 1'"}},
    "required": ["expression"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# 工具 2：search_knowledge_base（复用 RAG 检索，把 RAG 变成 Agent 可用的工具）
# --------------------------------------------------------------------------- #
SEARCH_KB_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "检索关键词或问题"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def build_local_tools(rag) -> list[Tool]:
    async def search_knowledge_base(query: str, top_k: int = 3) -> dict:
        hits = await rag.retrieve(query, top_k)
        return {"query": query, "hits": [{"doc_id": h["doc_id"], "chunk_id": h["chunk_id"],
                                          "score": h["score"], "text": h["text"][:400]} for h in hits]}

    return [
        Tool("calculator", "计算数学表达式，支持 + - * / // % **。", CALCULATOR_SCHEMA, calculator),
        Tool("search_knowledge_base", "在已上传的知识库文档中检索与问题相关的片段。",
             SEARCH_KB_SCHEMA, search_knowledge_base),
    ]

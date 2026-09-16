from __future__ import annotations

import ast
import operator
import re
from abc import ABC, abstractmethod
from typing import Any, Dict


class Tool(ABC):

    name: str
    description: str
    parameters: Dict[str, Any]

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.FloorDiv: operator.floordiv,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量类型: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式节点: {ast.dump(node)}")


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "计算一个数学表达式的结果。支持 + - * / // % ** 和括号，例如 '(12+8)*3/2'。"
        "当需要做算术、比例、预算换算等计算时调用它，而不要自己心算。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "只包含数字和 + - * / // % ** () 的算术表达式",
            }
        },
        "required": ["expression"],
    }

    def run(self, expression: str) -> str:
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree.body)
            return str(result)
        except Exception as exc:  # noqa: BLE001
            return f"计算出错: {exc}"


_MOCK_KNOWLEDGE_BASE = {
    "eiffel tower": "埃菲尔铁塔位于法国巴黎，高约330米，1889年建成，是巴黎的地标建筑。",
    "python": "Python 是一种解释型、通用编程语言，以简洁易读的语法著称，广泛用于数据科学与 Web 开发。",
    "great wall": "长城是中国古代的军事防御工程，总长超过2万公里，是世界文化遗产之一。",
    "tokyo weather": "东京属于温带海洋性气候，四季分明，春秋最适合旅行。",
    "paris budget hotel": "巴黎经济型酒店平均每晚价格约在 80-150 欧元之间。",
    "react agent": "ReAct 是一种让大模型交替进行推理(Thought)和行动(Action)的提示范式，"
    "由 Yao et al. 2022 提出，常用于构建可以调用外部工具的智能体。",
}


class SearchTool(Tool):
    name = "search"
    description = (
        "在知识库/网络中检索一个查询词，返回相关的简要信息。"
        "当需要查找事实、地点、最新信息或者不确定的知识时调用它。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "要搜索的关键词或问题"}
        },
        "required": ["query"],
    }

    def run(self, query: str) -> str:
        q = query.strip().lower()
        for key, value in _MOCK_KNOWLEDGE_BASE.items():
            if key in q or q in key:
                return value
        best_key, best_score = None, 0
        q_tokens = set(re.findall(r"[a-z0-9]+", q))
        for key, value in _MOCK_KNOWLEDGE_BASE.items():
            score = len(q_tokens & set(re.findall(r"[a-z0-9]+", key)))
            if score > best_score:
                best_key, best_score = key, score
        if best_key:
            return _MOCK_KNOWLEDGE_BASE[best_key]
        return f"没有在知识库中找到关于「{query}」的直接结果（这是一个离线 mock 搜索工具，可替换为真实搜索 API）。"



_MOCK_WEATHER = {
    "tokyo": "20°C，多云，适合户外活动",
    "paris": "17°C，小雨，建议带伞",
    "beijing": "24°C，晴，紫外线较强",
    "new york": "22°C，晴转多云",
}


class WeatherTool(Tool):
    name = "weather"
    description = "查询某个城市当前/近期的天气概况，用于旅行规划等场景。"
    parameters = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称，例如 'Tokyo'"}
        },
        "required": ["city"],
    }

    def run(self, city: str) -> str:
        key = city.strip().lower()
        if key in _MOCK_WEATHER:
            return f"{city}: {_MOCK_WEATHER[key]}"
        return f"{city}: 暂无天气数据（mock 工具，可替换为真实天气 API）。"


def default_toolset() -> list[Tool]:
    return [CalculatorTool(), SearchTool(), WeatherTool()]

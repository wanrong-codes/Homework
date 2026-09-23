from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .function_calling_agent import FunctionCallingAgent
from .llm_client import LLMClient, MockLLMClient
from .memory import MemoryStore
from .tools import Tool

MAX_REVISIONS = 2


class WorkflowState(TypedDict, total=False):
    task: str
    user_id: str
    plan: List[str]
    current_step: int
    step_result: str
    step_history: List[Dict[str, str]]
    revision_count: int
    feedback: str
    final_answer: str



def _extract_json(text: str) -> Optional[Any]:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    raw = fence.group(1) if fence else text
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
        return None


def _heuristic_plan(task: str) -> List[str]:
    parts = [s.strip() for s in re.split(r"[；;。\n]|然后|接着|并且|并", task) if s.strip()]
    if len(parts) <= 1:
        return [
            f"收集完成「{task}」所需要的信息",
            f"基于收集到的信息，完成：{task}",
        ]
    return parts[:4]


# ---------------------------------------------------------------------------
# Planner Agent
# ---------------------------------------------------------------------------
def build_planner_node(llm_client: LLMClient):
    def planner_node(state: WorkflowState) -> WorkflowState:
        task = state["task"]

        if isinstance(llm_client, MockLLMClient):
            plan = _heuristic_plan(task)
        else:
            prompt = (
                "你是一个任务规划者(Planner)。请把下面的用户任务拆分成 2-4 个具体、"
                "可以独立执行的子步骤。只输出一个 JSON 字符串数组，不要输出任何其他文字、"
                "不要用 markdown 代码块。\n"
                f"用户任务：{task}"
            )
            raw = llm_client.complete(prompt)
            parsed = _extract_json(raw)
            plan = [str(s) for s in parsed] if isinstance(parsed, list) and parsed else _heuristic_plan(task)

        return {
            **state,
            "plan": plan,
            "current_step": 0,
            "step_history": [],
            "revision_count": 0,
            "feedback": "",
        }

    return planner_node


# ---------------------------------------------------------------------------
# Worker Agent —— 复用 FunctionCallingAgent，工具里可以混入 MCP 工具
# ---------------------------------------------------------------------------
def build_worker_node(llm_client: LLMClient, tools: List[Tool], memory: Optional[MemoryStore] = None):
    system_prompt = (
        "你是一个执行者(Worker)。你会收到当前需要完成的子任务，必要时调用工具"
        "（包括通过 MCP 协议提供的工具）来完成它，然后给出简洁、明确的执行结果。"
        "如果收到了审查者(Critic)的修改意见，请针对意见改进你的回答，而不是重复之前的答案。"
    )
    worker_agent = FunctionCallingAgent(llm_client, tools, system_prompt=system_prompt)

    def worker_node(state: WorkflowState) -> WorkflowState:
        step = state["plan"][state["current_step"]]
        prompt = f"当前子任务：{step}"

        if state.get("feedback"):
            prompt += f"\n\nCritic 的修改意见（请据此改进你上一次的结果）：{state['feedback']}"

        if memory is not None and state.get("user_id"):
            pref_text = memory.format_for_prompt(state["user_id"])
            if pref_text:
                prompt = f"{pref_text}\n\n{prompt}"

        result = worker_agent.run(prompt, verbose=False)
        return {**state, "step_result": result}

    return worker_node


# ---------------------------------------------------------------------------
# Critic Agent
# ---------------------------------------------------------------------------
def build_critic_node(llm_client: LLMClient):
    def critic_node(state: WorkflowState) -> WorkflowState:
        step = state["plan"][state["current_step"]]
        result = state["step_result"]
        revision_count = state.get("revision_count", 0)

        if isinstance(llm_client, MockLLMClient):
            verdict, feedback = "APPROVE", ""
        else:
            prompt = (
                "你是一个审查者(Critic)。请判断 Worker 的执行结果是否完整、正确地"
                "完成了子任务，是否存在遗漏或错误。\n"
                f"子任务：{step}\n执行结果：{result}\n\n"
                '只输出一个 JSON 对象，不要输出其他文字：\n'
                '完成得好就输出 {"verdict": "APPROVE"}；\n'
                '需要修改就输出 {"verdict": "REVISE", "feedback": "具体、可执行的修改意见"}'
            )
            raw = llm_client.complete(prompt)
            parsed = _extract_json(raw)
            if isinstance(parsed, dict):
                verdict = parsed.get("verdict", "APPROVE")
                feedback = parsed.get("feedback", "")
            else:
                verdict, feedback = "APPROVE", ""

        if verdict == "REVISE" and revision_count < MAX_REVISIONS:
            return {**state, "feedback": feedback, "revision_count": revision_count + 1}

        
        history = list(state.get("step_history", []))
        history.append({"step": step, "result": result})
        return {
            **state,
            "step_history": history,
            "current_step": state["current_step"] + 1,
            "revision_count": 0,
            "feedback": "",
        }

    return critic_node


def _route_after_critic(state: WorkflowState) -> str:
    if state.get("feedback"):
        return "worker"  
    if state["current_step"] < len(state["plan"]):
        return "worker"  
    return "finalize"  


def build_finalize_node():
    def finalize_node(state: WorkflowState) -> WorkflowState:
        lines = [f"{i + 1}. {h['step']}\n   -> {h['result']}" for i, h in enumerate(state.get("step_history", []))]
        final_answer = f"任务「{state['task']}」已完成，各步骤结果如下：\n\n" + "\n".join(lines)
        return {**state, "final_answer": final_answer}

    return finalize_node


# ---------------------------------------------------------------------------
# 组装 LangGraph 图：Planner -> Worker -> Critic -> (回到 Worker | Finalize)
# ---------------------------------------------------------------------------
def build_multi_agent_graph(
    llm_client: LLMClient,
    tools: List[Tool],
    memory: Optional[MemoryStore] = None,
):
    graph = StateGraph(WorkflowState)
    graph.add_node("planner", build_planner_node(llm_client))
    graph.add_node("worker", build_worker_node(llm_client, tools, memory))
    graph.add_node("critic", build_critic_node(llm_client))
    graph.add_node("finalize", build_finalize_node())

    graph.set_entry_point("planner")
    graph.add_edge("planner", "worker")
    graph.add_edge("worker", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"worker": "worker", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


def run_multi_agent_task(
    task: str,
    llm_client: LLMClient,
    tools: List[Tool],
    memory: Optional[MemoryStore] = None,
    user_id: str = "default_user",
) -> Dict[str, Any]:
    app = build_multi_agent_graph(llm_client, tools, memory)
    return app.invoke({"task": task, "user_id": user_id})

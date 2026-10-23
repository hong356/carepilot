"""工具注册与统一执行入口。

工具结果是唯一的「事实来源」：金额、政策编号、工单状态都只能来自这里，
模型输出会由护栏层与此处返回的事实做交叉校验。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from ..rag import PolicyStore
from ..store import OrderStore


@dataclass
class ToolContext:
    now: datetime
    orders: OrderStore
    policies: PolicyStore
    session_id: str = ""
    turn: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


ToolHandler = Callable[[dict[str, Any], ToolContext], dict[str, Any]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, raw_args: str | dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        started = time.perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            return {
                "ok": False,
                "error": "UNKNOWN_TOOL",
                "message": f"不存在名为 {name} 的工具",
                "latency_ms": 0.0,
            }

        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as exc:
                return {
                    "ok": False,
                    "error": "BAD_ARGS",
                    "message": f"参数 JSON 解析失败：{exc}",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                }
        else:
            args = raw_args or {}

        missing = [key for key in tool.parameters.get("required", []) if key not in args]
        if missing:
            return {
                "ok": False,
                "error": "MISSING_ARGS",
                "message": f"缺少必填参数：{', '.join(missing)}",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }

        try:
            result = tool.handler(args, ctx)
        except Exception as exc:  # 工具异常不允许打断整个 Agent
            result = {
                "ok": False,
                "error": "TOOL_ERROR",
                "message": f"工具执行失败：{type(exc).__name__}: {exc}",
            }
        result.setdefault("tool", name)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return result


def build_registry() -> ToolRegistry:
    from . import handoff, orders, policy, rules

    registry = ToolRegistry()
    registry.register(orders.QUERY_ORDER)
    registry.register(orders.QUERY_LOGISTICS)
    registry.register(rules.CHECK_ELIGIBILITY)
    registry.register(rules.CREATE_TICKET)
    registry.register(policy.SEARCH_POLICY)
    registry.register(handoff.TRANSFER_TO_HUMAN)
    return registry

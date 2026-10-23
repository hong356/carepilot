"""人工转接工具。"""
from __future__ import annotations

from typing import Any

from .ids import next_id
from .registry import Tool, ToolContext

P0_KEYWORDS = ("投诉", "12315", "曝光", "律师", "欺诈", "人身", "食品安全")


def _transfer(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    order_id = str(args.get("order_id", "")).strip()
    order = ctx.orders.get(order_id)
    user = ctx.orders.user(order.get("user_id")) if order else {}
    reason = str(args.get("reason", ""))[:200]
    summary = str(args.get("summary", ""))[:300]
    text = reason + summary

    if any(keyword in text for keyword in P0_KEYWORDS):
        priority, sla_minutes, position = "P0", 5, 1
    elif user.get("risk_level") == "high" or user.get("tier") == "PLUS会员":
        priority, sla_minutes, position = "P1", 15, 2
    else:
        priority, sla_minutes, position = "P2", 30, 3

    ticket_id = next_id("CS", ctx.now)
    return {
        "ok": True,
        "policy_ids": ["P-COM-001"],
        "data": {
            "ticket_id": ticket_id,
            "order_id": order_id or None,
            "priority": priority,
            "queue_position": position,
            "sla_minutes": sla_minutes,
            "queue": "售后专席",
            "reason": reason,
            "summary": summary,
            "message": f"已进入人工队列，优先级 {priority}，预计 {sla_minutes} 分钟内响应",
        },
    }


TRANSFER_TO_HUMAN = Tool(
    name="transfer_to_human",
    description=(
        "转接人工客服并创建工单。用户明确要求人工、情绪激烈、投诉/12315、涉及人身安全、"
        "或智能客服无法处理时调用。调用前应尽量收集订单号与诉求摘要。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "订单号，没有可传空字符串"},
            "reason": {"type": "string", "description": "转接原因"},
            "summary": {"type": "string", "description": "给人工客服的交接摘要"},
        },
        "required": ["order_id", "reason", "summary"],
    },
    handler=_transfer,
)

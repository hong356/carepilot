"""订单与物流查询工具（只读）。"""
from __future__ import annotations

from typing import Any

from ..config import parse_dt
from .registry import Tool, ToolContext


def _hours(now, then) -> float:
    return max(0.0, (now - then).total_seconds() / 3600)


def _query_order(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    order_id = str(args.get("order_id", "")).strip()
    order = ctx.orders.get(order_id)
    if not order:
        return {
            "ok": False,
            "error": "ORDER_NOT_FOUND",
            "message": "未查询到该订单，请核对订单号后重试",
            "data": {"order_id": order_id},
        }
    user = ctx.orders.user(order.get("user_id"))
    data = ctx.orders.public_order(order)
    delivered = parse_dt(order.get("delivered_at"))
    data["days_since_delivered"] = (
        (ctx.now.date() - delivered.date()).days if delivered else None
    )
    data["member_tier"] = user.get("tier")
    return {"ok": True, "data": data}


def _query_logistics(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    order_id = str(args.get("order_id", "")).strip()
    order = ctx.orders.get(order_id)
    if not order:
        return {
            "ok": False,
            "error": "ORDER_NOT_FOUND",
            "message": "未查询到该订单，无法查询物流",
            "data": {"order_id": order_id},
        }

    status = order.get("status")
    events = list(order.get("logistics_events") or [])
    if not events and order.get("delivered_at"):
        events = [
            {
                "time": order["delivered_at"],
                "status": "已签收",
                "desc": "包裹已签收，签收人：本人",
            }
        ]

    last_update = events[-1]["time"] if events else None
    last_dt = parse_dt(last_update) or parse_dt(order.get("shipped_at"))
    stale_hours = (
        round(_hours(ctx.now, last_dt), 1) if last_dt and status == "运输中" else 0.0
    )
    shipped = parse_dt(order.get("shipped_at"))
    days_since_shipped = (
        (ctx.now.date() - shipped.date()).days if shipped else None
    )

    message = ""
    if status == "待发货":
        message = "商家尚未发货，暂无物流轨迹"
    elif not events:
        message = "暂无物流轨迹，可能尚未揽收"

    return {
        "ok": True,
        "data": {
            "order_id": order["order_id"],
            "status": status,
            "carrier": order.get("carrier"),
            "tracking_no": order.get("tracking_no"),
            "events": events,
            "last_update": last_update,
            "stale_hours": stale_hours,
            "days_since_shipped": days_since_shipped,
            "delivered_at": order.get("delivered_at"),
            "message": message,
        },
    }


QUERY_ORDER = Tool(
    name="query_order",
    description=(
        "按订单号查询订单详情，包括商品、金额、订单状态、签收时间、发票状态、售后记录、会员等级。"
        "处理任何售后诉求前，如果用户提供了订单号，应先调用本工具。订单号形如 TB2026102200101。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "订单号，例如 TB2026102200101"}
        },
        "required": ["order_id"],
    },
    handler=_query_order,
)


QUERY_LOGISTICS = Tool(
    name="query_logistics",
    description=(
        "查询订单物流轨迹、承运商、运单号、停滞时长和发货天数。用户询问物流进度、"
        "未收到货、物流延迟时调用。应根据返回的 stale_hours 判断是否超过 48 小时异常阈值。"
    ),
    parameters={
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "订单号"}},
        "required": ["order_id"],
    },
    handler=_query_logistics,
)

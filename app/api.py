"""FastAPI 服务：JSON 接口 + SSE 流式接口 + 单页演示界面。"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .agent import CareAgent
from .config import business_now, load_settings, parse_dt
from .trace import read_trace

settings = load_settings()
agent = CareAgent(settings)
WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="CarePilot", version=__version__, docs_url="/docs")


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    user_id: str = "U1001"


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": __version__,
        "mode": settings.mode,
        "model": settings.chat_model if not settings.offline else "mock",
        "now": business_now().isoformat(),
        "orders": len(agent.orders),
        "policy_chunks": len(agent.policies.chunks),
        "max_steps": settings.max_steps,
    }


@app.post("/api/chat")
async def chat(payload: ChatIn) -> dict[str, Any]:
    session = agent.sessions.get_or_create(payload.session_id, payload.user_id)
    result = await asyncio.to_thread(agent.run, session, payload.message.strip())
    data = result.as_dict()
    data["session_id"] = session.id
    return data


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatIn) -> StreamingResponse:
    session = agent.sessions.get_or_create(payload.session_id, payload.user_id)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

    def emit(event: str, data: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (event, data))

    def work() -> None:
        try:
            agent.run(session, payload.message.strip(), emit=emit)
        except Exception as exc:  # 保证前端一定收到终止事件
            emit("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            emit("done", {})

    threading.Thread(target=work, daemon=True, name=f"agent-{session.id}").start()

    async def generator():
        while True:
            event, data = await queue.get()
            if event == "done":
                break
            yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/session/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    session = agent.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {
        "session_id": session.id,
        "user_id": session.user_id,
        "messages": session.messages,
        "summary": session.summary,
        "turn": session.turn,
        "last_trace_id": session.last_trace_id,
    }


@app.get("/api/trace/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    events = read_trace(settings.trace_dir, trace_id)
    if not events:
        raise HTTPException(status_code=404, detail="trace not found")
    return {"trace_id": trace_id, "events": events}


def _question_for(order: dict[str, Any], days: int | None) -> str:
    oid = order["order_id"]
    category = order["items"][0].get("category", "")
    if order["status"] == "待发货":
        return f"订单 {oid} 还没发货，我不想要了，能直接退款吗？"
    if order["status"] == "运输中":
        return f"订单 {oid} 物流好几天没更新了，帮我看看怎么回事？"
    if order["status"] == "已取消":
        return f"订单 {oid} 取消了，退款还没到账，帮我查一下"
    if category == "生鲜":
        return f"订单 {oid} 送到的时候有死蟹，能退吗？"
    if category in ("手机", "电脑"):
        return f"订单 {oid} 已经激活了，还能退吗？"
    if category == "美妆":
        return f"订单 {oid} 拆封了还能退吗？"
    if category == "虚拟商品":
        return f"订单 {oid} 是虚拟商品，我不想要了能退吗？"
    current = order.get("current_price")
    if current is not None and current < order.get("paid_amount", 0):
        return f"订单 {oid} 降价了，能申请价保吗？"
    if days is not None:
        return f"订单 {oid} 签收 {days} 天了，我不想要了，能退吗？"
    return f"订单 {oid} 有什么问题，能帮我看看吗？"


def _tag_for(order: dict[str, Any]) -> str:
    if order.get("after_sale"):
        return "处理中"
    if order["status"] == "运输中":
        return "物流异常"
    if order["status"] == "待发货":
        return "未发货"
    if order.get("paid_amount", 0) > 500:
        return "大额"
    if order["items"][0].get("category") == "生鲜":
        return "生鲜"
    if order["items"][0].get("category") == "虚拟商品":
        return "虚拟"
    return "已签收"


@app.get("/api/demo/orders")
async def demo_orders(user_id: str = "U1001") -> dict[str, Any]:
    now = business_now()
    orders = []
    for order in agent.orders.orders_of_user(user_id):
        delivered = parse_dt(order.get("delivered_at"))
        days = (now.date() - delivered.date()).days if delivered else None
        orders.append(
            {
                "order_id": order["order_id"],
                "status": order["status"],
                "title": order["items"][0]["title"],
                "paid": order["paid_amount"],
                "days_since_delivered": days,
                "tag": _tag_for(order),
                "question": _question_for(order, days),
            }
        )
    return {"user_id": user_id, "orders": orders}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", media_type="text/html; charset=utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)

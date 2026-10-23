from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["mode"] == "offline"
    assert data["orders"] >= 10


def test_chat_endpoint():
    res = client.post(
        "/api/chat",
        json={
            "message": "订单 TB2026102200101 签收两天了，不想要了，能退吗？",
            "user_id": "U1001",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "499" in data["answer"]
    assert data["session_id"]
    assert [run["name"] for run in data["tool_runs"]][:1] == ["query_order"]
    assert data["metrics"]["steps"] >= 2


def test_chat_stream():
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "message": "订单 TB2026101201212 降价了，能申请价保吗？",
            "user_id": "U1001",
        },
    ) as res:
        assert res.status_code == 200
        body = "".join(res.iter_text())

    assert "event: tool_start" in body
    assert "event: token" in body
    assert "event: final" in body
    assert "P-PRI-001" in body


def test_demo_orders():
    res = client.get("/api/demo/orders?user_id=U1002")
    assert res.status_code == 200
    orders = res.json()["orders"]
    assert orders
    assert all("question" in item for item in orders)


def test_unknown_session_404():
    res = client.get("/api/session/does-not-exist")
    assert res.status_code == 404

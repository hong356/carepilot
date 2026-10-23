from __future__ import annotations

import pytest

from app.tools.registry import ToolContext, build_registry
from app.tools.rules import canonical_reason, canonical_request_type, evaluate_eligibility


CASES = [
    # order, user, request_type, reason, eligible, max_refund, requires_human
    ("TB2026102200101", "U1001", "退货退款", "七天无理由", True, 499.0, False),
    ("TB2026102200101", "U1001", "仅退款", "七天无理由", False, 0.0, False),
    ("TB2026101400202", "U1001", "退货退款", "七天无理由", False, 0.0, False),
    ("JD2026102300303", "U1002", "退货退款", "质量问题", True, 128.0, False),
    ("JD2026102300303", "U1002", "退货退款", "七天无理由", False, 0.0, False),
    ("PDD2026102200404", "U1002", "仅退款", "未收到货", True, 2680.0, True),
    ("PDD2026102200404", "U1002", "退货退款", "未收到货", False, 0.0, False),
    ("JD2026102100505", "U1001", "退货退款", "七天无理由", False, 0.0, False),
    ("JD2026102100505", "U1001", "维修", "质量问题", True, 0.0, True),
    ("ZTO2026102000606", "U1001", "物流", "物流延迟", True, 10.0, False),
    ("TB2026101900707", "U1002", "退货退款", "七天无理由", False, 0.0, False),
    ("TB2026101900707", "U1002", "退货退款", "质量问题", True, 199.0, False),
    ("TB2026093000808", "U1001", "退货退款", "质量问题", False, 0.0, False),
    ("TB2026093000808", "U1001", "维修", "质量问题", True, 0.0, False),
    ("JD2026102200909", "U1001", "退货退款", "七天无理由", False, 0.0, False),
    ("TB2026102001010", "U1003", "退货退款", "质量问题", True, 359.0, True),
    ("TB2026101201212", "U1001", "价保", "价格保护", True, 700.0, True),
    ("TB2026102301313", "U1002", "退货退款", "七天无理由", True, 259.0, False),
    ("JD2026102401414", "U1001", "仅退款", "未收到货", True, 329.0, False),
    ("TB2026102101515", "U1003", "退货退款", "质量问题", False, 0.0, False),
    ("TB2026091001616", "U1001", "退货退款", "七天无理由", False, 0.0, False),
    ("PDD2026101601111", "U1002", "退货退款", "七天无理由", False, 0.0, False),
    ("PDD2026101601111", "U1002", "退货退款", "质量问题", True, 89.0, False),
]


@pytest.mark.parametrize(
    "order_id,user_id,request_type,reason,eligible,max_refund,requires_human",
    CASES,
    ids=[f"{c[0]}-{c[2]}-{c[3]}" for c in CASES],
)
def test_eligibility_matrix(
    order_id, user_id, request_type, reason, eligible, max_refund, requires_human, store, now
):
    order = store.get(order_id)
    user = store.user(user_id)
    result = evaluate_eligibility(order, user, request_type, reason, now, auto_limit=500.0)

    assert result["eligible"] is eligible, result["rules"]
    assert result["max_refund"] == pytest.approx(max_refund)
    assert result["requires_human"] is requires_human, result["rules"]
    if eligible:
        assert result["decision"] in {"可以受理", "可以受理，需人工审核"}
        assert result["policy_ids"]
    else:
        assert result["decision"] in {"暂不可受理", "需人工核实"}


def test_canonical_reason_and_type():
    assert canonical_reason("7天无理由不想要了") == "七天无理由"
    assert canonical_reason("收到的鞋子开胶") == "质量问题"
    assert canonical_reason("货不对板") == "描述不符"
    assert canonical_reason("快递一直不更新") == "物流延迟"
    assert canonical_request_type("只退款不退货") == "仅退款"
    assert canonical_request_type("申请保价") == "价保"


def _ctx(store, policies, now):
    return ToolContext(now=now, orders=store, policies=policies, meta={"auto_limit": 500.0})


def test_ticket_rejects_ineligible(store, policies, now):
    registry = build_registry()
    result = registry.execute(
        "create_after_sale_ticket",
        {
            "order_id": "TB2026101400202",
            "request_type": "退货退款",
            "reason": "七天无理由",
        },
        _ctx(store, policies, now),
    )
    assert result["ok"] is False
    assert result["error"] == "NOT_ELIGIBLE"
    assert store.get("TB2026101400202")["after_sale"] is None


def test_ticket_auto_approved(store, policies, now):
    registry = build_registry()
    result = registry.execute(
        "create_after_sale_ticket",
        {
            "order_id": "TB2026102301313",
            "request_type": "退货退款",
            "reason": "七天无理由",
            "note": "用户主动申请",
        },
        _ctx(store, policies, now),
    )
    data = result["data"]
    assert result["ok"] is True
    assert data["status"] == "auto_approved"
    assert data["approved_amount"] == pytest.approx(259.0)
    assert data["type"] == "退货退款"


def test_ticket_pending_for_large_amount(store, policies, now):
    registry = build_registry()
    result = registry.execute(
        "create_after_sale_ticket",
        {
            "order_id": "PDD2026102200404",
            "request_type": "仅退款",
            "reason": "未收到货",
        },
        _ctx(store, policies, now),
    )
    data = result["data"]
    assert result["ok"] is True
    assert data["status"] == "pending_human"
    assert data["approved_amount"] == 0.0
    assert data["max_refund"] == pytest.approx(2680.0)


def test_ticket_idempotent(store, policies, now):
    registry = build_registry()
    args = {
        "order_id": "TB2026102301313",
        "request_type": "退货退款",
        "reason": "七天无理由",
    }
    first = registry.execute("create_after_sale_ticket", args, _ctx(store, policies, now))
    second = registry.execute("create_after_sale_ticket", args, _ctx(store, policies, now))
    assert second["ok"] is True
    assert second["data"]["duplicated"] is True
    assert second["data"]["ticket_id"] == first["data"]["ticket_id"]


def test_existing_ticket_blocks_check(store, now):
    order = store.get("TB2026102101515")
    result = evaluate_eligibility(
        order, store.user("U1003"), "退货退款", "质量问题", now, 500.0
    )
    assert result["eligible"] is False
    assert any(rule["code"] == "R-EXISTING-TICKET" for rule in result["rules"])

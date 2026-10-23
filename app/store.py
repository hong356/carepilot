"""订单与用户数据访问层。演示数据常驻内存，写操作直接反映在会话内。"""
from __future__ import annotations

import copy
import json
from pathlib import Path


class OrderStore:
    def __init__(self, orders: list[dict], users: list[dict]) -> None:
        self._orders = {o["order_id"].upper(): o for o in orders}
        self._users = {u["user_id"]: u for u in users}

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> "OrderStore":
        orders = json.loads((Path(data_dir) / "orders.json").read_text(encoding="utf-8"))
        users = json.loads((Path(data_dir) / "users.json").read_text(encoding="utf-8"))
        return cls(orders, users)

    def get(self, order_id: str | None) -> dict | None:
        if not order_id:
            return None
        return self._orders.get(order_id.strip().upper())

    def user(self, user_id: str | None) -> dict:
        return self._users.get(user_id or "", {"user_id": user_id, "tier": "普通会员", "risk_level": "normal"})

    def __len__(self) -> int:
        return len(self._orders)

    def orders_of_user(self, user_id: str) -> list[dict]:
        return [o for o in self._orders.values() if o.get("user_id") == user_id]

    def reset(self, data_dir: Path) -> None:
        orders = json.loads((Path(data_dir) / "orders.json").read_text(encoding="utf-8"))
        users = json.loads((Path(data_dir) / "users.json").read_text(encoding="utf-8"))
        self._orders = {o["order_id"].upper(): o for o in orders}
        self._users = {u["user_id"]: u for u in users}

    @staticmethod
    def public_order(order: dict) -> dict:
        """给模型看的订单视图：不含风控标签等敏感字段（最小权限）。"""
        return copy.deepcopy({k: v for k, v in order.items() if k != "audit"})

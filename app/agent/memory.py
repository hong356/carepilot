"""会话记忆：短期窗口 + 滚动摘要 + 结构化事实。

事实（facts）是护栏跨轮次判断的依据，例如上一轮已授权的最大退款金额。
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def new_session_id() -> str:
    return uuid.uuid4().hex[:10]


@dataclass
class Session:
    id: str
    user_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    turn: int = 0
    last_trace_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "messages": self.messages,
            "summary": self.summary,
            "facts": self.facts,
            "turn": self.turn,
            "last_trace_id": self.last_trace_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(**data)


class SessionStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.path and self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._sessions = {sid: Session.from_dict(item) for sid, item in raw.items()}
            except (json.JSONDecodeError, TypeError):
                self._sessions = {}

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {sid: s.to_dict() for sid, s in self._sessions.items()}
            self.path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None, user_id: str) -> Session:
        sid = session_id or new_session_id()
        with self._lock:
            session = self._sessions.get(sid)
            if session is None:
                session = Session(id=sid, user_id=user_id or "U1001")
                self._sessions[sid] = session
            elif user_id:
                session.user_id = user_id
        return session

    def reset(self) -> None:
        self._sessions.clear()

    def all(self) -> list[Session]:
        return list(self._sessions.values())

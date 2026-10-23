"""轻量级结构化追踪：每次会话写入 JSONL，可离线复盘每一步决策。"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


class Tracer:
    def __init__(self, trace_dir: Path, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or new_trace_id()
        self.dir = Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{self.trace_id}.jsonl"
        self._lock = threading.Lock()
        self._start = time.perf_counter()

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "elapsed_ms": round((time.perf_counter() - self._start) * 1000, 1),
            "trace_id": self.trace_id,
            "event": event,
            **fields,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_trace(trace_dir: Path, trace_id: str) -> list[dict[str, Any]]:
    path = Path(trace_dir) / f"{trace_id}.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


ProgressFn = Callable[[str, dict[str, Any]], None]

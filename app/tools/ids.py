"""工单号生成。"""
from __future__ import annotations

import itertools
import threading
from datetime import datetime

_seq = itertools.count(1)
_lock = threading.Lock()


def next_id(prefix: str, now: datetime) -> str:
    with _lock:
        n = next(_seq)
    return f"{prefix}{now:%Y%m%d}{n:04d}"

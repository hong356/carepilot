"""全局配置。

业务时间通过 DEMO_NOW 固定，保证评测可复现（时间冻结）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

TZ = timezone(timedelta(hours=8))
DEFAULT_NOW = "2026-10-24T10:00:00+08:00"

# 单位为「元 / 百万 tokens」。价格会调整，以 DeepSeek 官网为准，这里只用于成本估算。
PRICING: dict[str, tuple[float, float]] = {
    "deepseek-chat": (2.0, 3.0),
    "deepseek-reasoner": (4.0, 8.0),
}


def business_now() -> datetime:
    raw = os.getenv("DEMO_NOW", DEFAULT_NOW)
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    chat_model: str
    judge_model: str
    max_steps: int
    temperature: float
    timeout: float
    data_dir: Path
    runtime_dir: Path
    trace_dir: Path
    offline: bool
    max_history_messages: int
    auto_approve_limit: float

    @property
    def mode(self) -> str:
        return "offline" if self.offline else "deepseek"


def load_settings() -> Settings:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    override = os.getenv("CAREPILOT_OFFLINE", "").strip().lower()
    placeholder = api_key in {"", "sk-your-key-here", "your-key-here"}
    offline = override in {"1", "true", "yes"} or placeholder
    runtime_dir = Path(os.getenv("CAREPILOT_RUNTIME", ROOT / "runtime"))
    return Settings(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        chat_model=os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
        judge_model=os.getenv("DEEPSEEK_JUDGE_MODEL", "deepseek-reasoner"),
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "6")),
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0.2")),
        timeout=float(os.getenv("LLM_TIMEOUT", "60")),
        data_dir=ROOT / "data",
        runtime_dir=runtime_dir,
        trace_dir=runtime_dir / "traces",
        offline=offline,
        max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "12")),
        auto_approve_limit=500.0,
    )

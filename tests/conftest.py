from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["CAREPILOT_OFFLINE"] = "1"
os.environ.setdefault("CAREPILOT_RUNTIME", tempfile.mkdtemp(prefix="carepilot-test-"))
os.environ.setdefault("DEMO_NOW", "2026-10-24T10:00:00+08:00")

import pytest  # noqa: E402

from app.config import business_now, load_settings  # noqa: E402
from app.rag import PolicyStore  # noqa: E402
from app.store import OrderStore  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    base = load_settings()
    return replace(
        base,
        runtime_dir=tmp_path,
        trace_dir=tmp_path / "traces",
        offline=True,
        api_key="",
    )


@pytest.fixture
def now():
    return business_now()


@pytest.fixture
def store():
    return OrderStore.from_data_dir(ROOT / "data")


@pytest.fixture
def policies():
    return PolicyStore.load(ROOT / "data" / "policies")

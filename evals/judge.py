"""可选：用 deepseek-reasoner 做 LLM-as-a-Judge，对最终回复做 1-5 分质检。"""
from __future__ import annotations

import json
from typing import Any

from app.config import Settings
from app.llm import DeepSeekLLM, Usage

JUDGE_SYSTEM = """你是电商平台的售后质检专家。请评价客服回复是否准确、合规、有用。
只输出一个 JSON 对象，不要输出 Markdown 或额外解释：
{"score": 1-5, "accuracy": 1-5, "compliance": 1-5, "helpfulness": 1-5, "comment": "80 字以内的中文点评"}
评分标准：
- accuracy：是否符合订单事实与平台规则，金额是否正确
- compliance：是否存在越权承诺、虚假“已办理”、编造政策编号
- helpfulness：结论是否明确，是否给出可执行的下一步
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型未返回 JSON")
    return json.loads(text[start : end + 1])


def judge_case(settings: Settings, case: dict[str, Any], answer: str) -> tuple[dict[str, Any], Usage, str]:
    if settings.offline or not settings.api_key:
        return {"error": "offline 模式不执行 LLM 裁判"}, Usage(), settings.judge_model

    expected = case.get("expect", {})
    user_content = (
        f"用户诉求（共 {len(case.get('turns', []))} 轮）：\n"
        + "\n".join(f"- {turn}" for turn in case.get("turns", []))
        + f"\n\n系统期望要点：可受理={expected.get('eligible')}，"
        f"最大金额={expected.get('max_refund_gte') or expected.get('max_refund_lte')}，"
        f"需人工={expected.get('requires_human')}。\n"
        f"\n客服最终回复：\n{answer[:1500]}"
    )

    llm = DeepSeekLLM(settings)
    try:
        result = llm.chat(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            model=settings.judge_model,
        )
        data = _extract_json(result.message.get("content") or "")
        return data, result.usage, result.model
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}, Usage(), settings.judge_model

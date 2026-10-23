"""政策检索工具。"""
from __future__ import annotations

from typing import Any

from .registry import Tool, ToolContext


def _search(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    query = str(args.get("query", "")).strip()
    if not query:
        return {"ok": False, "error": "EMPTY_QUERY", "message": "检索词不能为空"}
    try:
        top_k = int(args.get("top_k", 4))
    except (TypeError, ValueError):
        top_k = 4
    top_k = max(1, min(top_k, 6))

    data = ctx.policies.search(query, top_k)
    if not data["results"]:
        return {
            "ok": True,
            "data": data,
            "policy_ids": [],
            "message": "未检索到相关政策，建议转人工确认",
        }
    return {"ok": True, "data": data, "policy_ids": data["policy_ids"]}


SEARCH_POLICY = Tool(
    name="search_policy",
    description=(
        "检索平台售后政策库，返回政策编号、标题、章节和原文片段。涉及规则解释、时效、"
        "类目例外、运费、赔付标准时必须调用，并在回复中用【P-XXX-000】引用编号。"
        "policy_id 已包含检索结果中，不要自行编写。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索词，例如：生鲜 七天无理由、运费险理赔"},
            "top_k": {"type": "integer", "description": "返回条数，1-6，默认 4"},
        },
        "required": ["query"],
    },
    handler=_search,
)

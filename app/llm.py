"""LLM 客户端封装。

- DeepSeekLLM：OpenAI 兼容协议，带指数退避重试、流式输出、token/成本统计。
- MockLLM：离线规则版，用于单元测试与无 Key 演示，接口完全一致。
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from .config import PRICING, Settings

TokenCallback = Callable[[str], None]


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    other = len(text) - cjk
    return max(1, int(cjk * 0.65 + other * 0.28))


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost(self, model: str) -> float:
        if model == "mock":
            return 0.0
        prompt_price, completion_price = PRICING.get(model, (2.0, 3.0))
        return round(
            self.prompt_tokens / 1_000_000 * prompt_price
            + self.completion_tokens / 1_000_000 * completion_price,
            6,
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass
class ChatResult:
    message: dict[str, Any]
    usage: Usage
    model: str
    latency_ms: float
    finish_reason: str = "stop"
    estimated_usage: bool = False


class DeepSeekLLM:
    name = "deepseek"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            max_retries=0,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        on_token: TokenCallback | None = None,
        model: str | None = None,
    ) -> ChatResult:
        model_name = model or self.settings.chat_model
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
        }
        # deepseek-reasoner 不支持采样参数，传了可能被拒绝
        if "reasoner" in model_name:
            kwargs.pop("temperature", None)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format
        if on_token:
            kwargs["stream"] = True
            return self._call_stream(kwargs, on_token)
        return self._call(kwargs)

    # ---------------------------------------------------------------- 非流式
    def _call(self, kwargs: dict[str, Any]) -> ChatResult:
        start = time.perf_counter()
        resp = self._with_retry(lambda: self.client.chat.completions.create(**kwargs))
        latency = (time.perf_counter() - start) * 1000
        choice = resp.choices[0]
        msg = choice.message
        message: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if getattr(msg, "tool_calls", None):
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        usage = Usage(
            getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
            getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
        )
        return ChatResult(
            message=message,
            usage=usage,
            model=kwargs["model"],
            latency_ms=round(latency, 1),
            finish_reason=choice.finish_reason or "stop",
        )

    # ------------------------------------------------------------------ 流式
    def _call_stream(self, kwargs: dict[str, Any], on_token: TokenCallback) -> ChatResult:
        start = time.perf_counter()
        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"

        def open_stream():
            return self.client.chat.completions.create(**kwargs)

        stream = self._with_retry(open_stream)
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is not None:
                if delta.content:
                    content_parts.append(delta.content)
                    on_token(delta.content)
                for tc in getattr(delta, "tool_calls", None) or []:
                    slot = tool_calls.setdefault(
                        tc.index,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["function"]["arguments"] += tc.function.arguments
            if choice.finish_reason:
                finish_reason = choice.finish_reason

        message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]

        estimated = "usage" not in locals()
        if estimated:
            prompt_text = json.dumps(kwargs["messages"], ensure_ascii=False)
            usage = Usage(estimate_tokens(prompt_text), estimate_tokens(message["content"]))

        return ChatResult(
            message=message,
            usage=usage,
            model=kwargs["model"],
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            finish_reason=finish_reason,
            estimated_usage=estimated,
        )

    # ------------------------------------------------------------------ 重试
    def _with_retry(self, fn):
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return fn()
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                last_error = exc
            except APIStatusError as exc:
                if exc.status_code not in (429, 500, 502, 503, 504):
                    raise
                last_error = exc
            if attempt < 2:
                time.sleep(min(1.5 * (2**attempt), 6) + random.random() * 0.3)
        assert last_error is not None
        raise last_error


class MockLLM:
    """离线模式：用规则模拟「计划工具 → 调用 → 总结」，用于 CI 和断网演示。"""

    name = "mock"

    ORDER_ID_RE = re.compile(r"\b([A-Z]{2,4}\d{10,})\b")
    FACT_RE = re.compile(
        r"- 订单 (\w+)：类型=([^；]+)；原因=([^；]+)；结论=([^；]+)；"
        r"最多可退 ¥([\d.]+)；需人工=(True|False)"
    )
    ACTION_HINTS = (
        "帮我提交", "帮我申请", "帮我办理", "帮我退", "帮我换", "给我退",
        "给我退款", "直接退", "现在就", "提交申请", "我要退款", "我要退货",
        "我要申请", "办理一下",
    )
    # 出现这些词说明用户在指代某个具体订单，可以复用上一轮提到的订单号
    ORDER_REFERENCE_HINTS = (
        "退", "换", "维修", "保修", "赔", "物流", "快递", "签收", "收到",
        "发票", "价保", "售后", "订单", "这笔", "发错", "漏发", "少件",
    )
    # 出现这些词且没有订单号时，按通用政策咨询处理，而不是套到上一笔订单上
    POLICY_QUESTION_HINTS = (
        "政策", "规则", "规定", "条款", "时效", "几天", "多久", "多少",
        "为什么", "支持", "如何", "标准",
    )

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        on_token: TokenCallback | None = None,
        model: str | None = None,
    ) -> ChatResult:
        start = time.perf_counter()
        user_text, tool_results = self._split_turn(messages)

        if response_format:
            content = json.dumps(
                {"score": 4, "accuracy": 4, "compliance": 4, "comment": "离线模拟评分"},
                ensure_ascii=False,
            )
            message = {"role": "assistant", "content": content}
        else:
            has_order = bool(self.ORDER_ID_RE.search(user_text.upper()))
            facts = self._facts_from_system(messages)
            followup = self._followup(user_text, facts) if facts and not has_order else None

            if followup and followup.get("answer"):
                message = {"role": "assistant", "content": followup["answer"]}
            else:
                plan = self._plan(user_text, tool_results, self._context_order_id(messages))
                if followup and followup.get("tool"):
                    plan = [followup["tool"]]
                if len(tool_results) < len(plan):
                    call = plan[len(tool_results)]
                    message = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call_mock_{len(tool_results)}",
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": json.dumps(call["args"], ensure_ascii=False),
                                },
                            }
                        ],
                    }
                else:
                    message = {
                        "role": "assistant",
                        "content": self._answer(user_text, tool_results),
                    }

        if on_token and message["content"]:
            for i in range(0, len(message["content"]), 12):
                on_token(message["content"][i : i + 12])
                time.sleep(0.01)

        prompt_text = json.dumps(messages, ensure_ascii=False)
        usage = Usage(
            estimate_tokens(prompt_text),
            estimate_tokens(message.get("content", ""))
            + sum(estimate_tokens(json.dumps(m, ensure_ascii=False)) for m in message.get("tool_calls", [])),
        )
        return ChatResult(
            message=message,
            usage=usage,
            model="mock",
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            estimated_usage=True,
        )

    # ------------------------------------------------------------ 多轮跟进
    @classmethod
    def _context_order_id(cls, messages: list[dict[str, Any]]) -> str:
        """从会话历史里找最近一次出现的订单号，供「退货」「查物流」这类指代使用。

        用户不一定每轮都复述订单号；只看当前这句话会导致反复索要订单号。
        """
        for msg in reversed(messages):
            role = msg.get("role")
            if role == "user":
                match = cls.ORDER_ID_RE.search(str(msg.get("content", "")).upper())
                if match:
                    return match.group(1)
            elif role == "tool":
                try:
                    payload = json.loads(msg.get("content", "{}"))
                except json.JSONDecodeError:
                    continue
                order_id = str((payload.get("data") or {}).get("order_id") or "")
                if cls.ORDER_ID_RE.search(order_id.upper()):
                    return order_id
        return ""

    @classmethod
    def _resolve_order_id(cls, text: str, context_order_id: str) -> str:
        """优先用本轮明确给出的订单号，否则在用户明确指向订单时复用上文的订单号。"""
        match = cls.ORDER_ID_RE.search(text.upper())
        if match:
            return match.group(1)
        if not context_order_id:
            return ""
        if not any(k in text for k in cls.ORDER_REFERENCE_HINTS):
            return ""
        if any(k in text for k in cls.POLICY_QUESTION_HINTS):
            # 「退货政策是几天」「运费险能赔多少」这类问题应查政策，而不是套用上一笔订单
            return ""
        return context_order_id

    @classmethod
    def _facts_from_system(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") != "system":
                continue
            for match in cls.FACT_RE.finditer(msg.get("content", "")):
                facts.append(
                    {
                        "order_id": match.group(1),
                        "request_type": match.group(2),
                        "reason": match.group(3),
                        "decision": match.group(4),
                        "max_refund": float(match.group(5)),
                        "requires_human": match.group(6) == "True",
                    }
                )
        return facts

    def _followup(self, text: str, facts: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not facts:
            return None
        fact = facts[-1]
        order_id = fact["order_id"]

        if any(hint in text for hint in self.ACTION_HINTS):
            return {
                "tool": {
                    "name": "create_after_sale_ticket",
                    "args": {
                        "order_id": order_id,
                        "request_type": fact["request_type"],
                        "reason": fact["reason"],
                        "note": "用户在多轮对话中确认办理",
                    },
                }
            }

        if any(k in text for k in ("多少", "金额", "退多少", "合计")):
            suffix = (
                "这笔申请需要人工审核，提交后预计 30 分钟内完成。"
                if fact["requires_human"]
                else "如需办理，我可以帮您提交申请。"
            )
            return {
                "answer": (
                    f"本会话已核实：订单 {order_id}（{fact['request_type']}）"
                    f"按规则最多可退 ¥{fact['max_refund']:.0f}。{suffix}"
                )
            }

        if any(k in text for k in ("那能退吗", "可以吗", "怎么办", "怎么处理")):
            return {
                "answer": (
                    f"本会话已核实：订单 {order_id} 的结论是「{fact['decision']}」。"
                    + (
                        "这笔申请需要人工审核。"
                        if fact["requires_human"]
                        else "如确认办理，我可以帮您提交申请。"
                    )
                )
            }
        return None

    # ------------------------------------------------------------ 计划生成
    def _split_turn(self, messages: list[dict[str, Any]]) -> tuple[str, list[tuple[str, dict]]]:
        last_user = ""
        start = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                start = i

        # 通过 tool_call_id 反查工具名（不依赖 tool 消息上的 name 字段）
        id_to_name: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                for call in msg.get("tool_calls") or []:
                    call_id = call.get("id", "")
                    if call_id:
                        id_to_name[call_id] = call.get("function", {}).get("name", "")

        results: list[tuple[str, dict]] = []
        for msg in messages[start:]:
            if msg.get("role") == "tool":
                try:
                    payload = json.loads(msg.get("content", "{}"))
                except json.JSONDecodeError:
                    payload = {}
                name = msg.get("name") or id_to_name.get(msg.get("tool_call_id", ""), "")
                results.append((name, payload))
        return last_user, results

    def _plan(
        self,
        text: str,
        results: list[tuple[str, dict]],
        context_order_id: str = "",
    ) -> list[dict[str, Any]]:
        order_id = self._resolve_order_id(text, context_order_id)

        if any(k in text for k in ("人工", "投诉", "12315", "曝光", "律师", "骗子")):
            return [
                {
                    "name": "transfer_to_human",
                    "args": {
                        "order_id": order_id,
                        "reason": "用户要求人工或投诉",
                        "summary": text[:100],
                    },
                }
            ]

        order_status = None
        logistics_data: dict[str, Any] | None = None
        for name, payload in results:
            if name == "query_order" and payload.get("ok"):
                order_status = (payload.get("data") or {}).get("status")
            if name == "query_logistics" and payload.get("ok"):
                logistics_data = payload.get("data") or {}

        if not order_id:
            if any(
                k in text
                for k in (
                    "政策", "规则", "几天", "多久", "怎么", "多少", "能不能", "可以吗",
                    "为什么", "支持", "规定", "条款", "如何", "时效",
                )
            ):
                return [{"name": "search_policy", "args": {"query": text, "top_k": 4}}]
            return []

        plan: list[dict[str, Any]] = [
            {"name": "query_order", "args": {"order_id": order_id}}
        ]

        wants_logistics = any(k in text for k in ("物流", "快递", "到哪", "没收到", "未收到", "延迟", "丢件"))
        wants_after_sale = any(
            k in text
            for k in (
                "退", "换", "维修", "赔", "质量", "坏", "破", "开胶", "划痕",
                "不符", "价保", "降价", "发票", "不想要", "售后",
            )
        )

        if wants_logistics:
            plan.append({"name": "query_logistics", "args": {"order_id": order_id}})

        if logistics_data and not wants_after_sale:
            stale = float(logistics_data.get("stale_hours") or 0)
            shipped = logistics_data.get("days_since_shipped")
            # 已签收的订单不存在“运输途中丢件”，不能仅按发货天数判定
            lost = order_status != "已签收" and shipped is not None and shipped >= 7
            if order_status == "已签收" and any(k in text for k in ("没收到", "未收到")):
                plan.append(
                    {
                        "name": "check_after_sale_eligibility",
                        "args": {
                            "order_id": order_id,
                            "request_type": "仅退款",
                            "reason": "未收到货",
                        },
                    }
                )
            elif stale >= 48 or lost:
                plan.append(
                    {
                        "name": "check_after_sale_eligibility",
                        "args": {
                            "order_id": order_id,
                            "request_type": "物流",
                            "reason": "物流延迟" if not lost else "未收到货",
                        },
                    }
                )

        if wants_after_sale:
            request_type = self._infer_request(text, order_status)
            reason = self._infer_reason(text)
            plan.append(
                {
                    "name": "check_after_sale_eligibility",
                    "args": {
                        "order_id": order_id,
                        "request_type": request_type,
                        "reason": reason,
                    },
                }
            )
            explicit_actions = (
                "帮我提交", "帮我申请", "帮我办理", "帮我退", "帮我换", "给我退",
                "给我退款", "直接退", "现在就", "提交申请", "我要退款", "我要退货",
                "我要申请", "办理一下",
            )
            if any(k in text for k in explicit_actions):
                plan.append(
                    {
                        "name": "create_after_sale_ticket",
                        "args": {
                            "order_id": order_id,
                            "request_type": request_type,
                            "reason": reason,
                            "note": text[:100],
                        },
                    }
                )
        return plan

    @staticmethod
    def _infer_request(text: str, order_status: str | None = None) -> str:
        if any(k in text for k in ("价保", "降价", "便宜了")):
            return "价保"
        if "发票" in text:
            return "发票"
        if "仅退款" in text or "只退款" in text:
            return "仅退款"
        if "换" in text:
            return "换货"
        if any(k in text for k in ("维修", "保修")):
            return "维修"
        if any(k in text for k in ("没收到", "未收到")):
            return "仅退款"
        if order_status in ("待发货", "运输中") and "退" in text:
            return "仅退款"
        return "退货退款"

    @staticmethod
    def _infer_reason(text: str) -> str:
        if any(k in text for k in ("质量", "坏", "破", "故障", "开胶", "划痕", "瑕疵", "变质", "死", "过敏")):
            return "质量问题"
        if any(k in text for k in ("不符", "货不对板", "假")):
            return "描述不符"
        if any(k in text for k in ("不想要", "不喜欢", "不合适", "无理由", "七天", "7天")):
            return "七天无理由"
        if any(k in text for k in ("没收到", "未收到")):
            return "未收到货"
        if any(k in text for k in ("降价", "便宜")):
            return "价格保护"
        if any(k in text for k in ("延迟", "慢", "停滞")):
            return "物流延迟"
        if "退" in text or "售后" in text:
            return "七天无理由"
        return "其他"

    # ------------------------------------------------------------ 答案生成
    def _answer(self, user_text: str, results: list[tuple[str, dict]]) -> str:
        if not results:
            return "您好，请提供订单号（例如 TB2026102200101），我帮您核实订单和售后条件。"

        for name, payload in reversed(results):
            data = payload.get("data") or {}
            if name == "transfer_to_human" and payload.get("ok"):
                return (
                    f"已为您转接人工客服，工单号 {data.get('ticket_id')}，"
                    f"当前排队第 {data.get('queue_position', 1)} 位，预计 {data.get('sla_minutes', 10)} 分钟内响应。"
                    "您不需要重复描述问题。"
                )
            if name == "create_after_sale_ticket" and payload.get("ok"):
                if data.get("duplicated"):
                    return (
                        f"该订单已有处理中的「{data.get('type')}」工单（{data.get('ticket_id')}），"
                        f"无需重复提交。当前状态：{data.get('status_label')}，{data.get('sla')}。"
                    )
                if data.get("status") == "auto_approved":
                    ticket_type = data.get("type") or data.get("request_type") or "售后"
                    return (
                        f"已为您提交{ticket_type}申请，工单号 {data.get('ticket_id')}，"
                        f"退款金额 ¥{data.get('approved_amount', 0):.0f}，"
                        f"{data.get('sla')}。"
                    )
                return (
                    f"已提交人工审核，工单号 {data.get('ticket_id')}，"
                    f"预计 {data.get('sla_minutes', 30)} 分钟内审核完成，结果会通过站内信通知您。"
                )
            if name == "check_after_sale_eligibility":
                if not payload.get("ok"):
                    return f"暂时无法完成判断：{payload.get('message', '请稍后重试')}。"
                return self._answer_eligibility(data)
            if name == "query_logistics" and payload.get("ok"):
                return self._answer_logistics(data)
            if name == "search_policy" and payload.get("ok"):
                return self._answer_policy(data)
            if name == "query_order" and not payload.get("ok"):
                return (
                    f"没有查询到订单 {data.get('order_id', '')}。"
                    "请核对订单号，或告诉我下单手机号后四位，我再帮您查一下。"
                )

        last_name, last_payload = results[-1]
        if last_name == "query_order" and last_payload.get("ok"):
            data = last_payload["data"]
            return (
                f"订单 {data.get('order_id')} 当前状态为「{data.get('status')}」，"
                f"实付 ¥{data.get('paid_amount', 0):.0f}。请问您想办理退货、换货还是查询物流？"
            )
        return "我需要再核实一下这笔订单的信息，请稍等。"

    def _answer_eligibility(self, data: dict[str, Any]) -> str:
        policies = "".join(f"【{pid}】" for pid in data.get("policy_ids", [])[:2])
        if data.get("eligible"):
            parts = [f"已核实，订单 {data['order_id']} 可以按「{data['request_type']}」流程处理。"]
            max_refund = float(data.get("max_refund") or 0)
            if max_refund > 0:
                if data["request_type"] == "价保":
                    parts.append(f"按规则最多可退差价 ¥{max_refund:.0f}。")
                elif data["request_type"] == "物流":
                    parts.append(f"按规则最多可补偿 ¥{max_refund:.0f}。")
                else:
                    parts.append(f"按规则最多可退 ¥{max_refund:.0f}。")
            if data.get("evidence_required"):
                parts.append("需要提供：" + "、".join(data["evidence_required"]) + "。")
            if data.get("requires_human"):
                parts.append("这笔申请需要人工审核，提交后预计 30 分钟内完成。")
            else:
                parts.append(data.get("next_step", "").rstrip("。") + "。")
            parts.append(f"政策依据{policies}。")
            return "".join(parts)

        reasons = "；".join(
            r["detail"] for r in data.get("rules", []) if not r.get("passed")
        ) or "不符合该售后类型的受理条件"
        suggestions = data.get("suggestions") or ["可以转人工进一步核实"]
        return (
            f"这笔订单暂时无法按「{data['request_type']}」办理：{reasons}。"
            f"可选方案：{suggestions[0]}。政策依据{policies}。"
        )

    def _answer_logistics(self, data: dict[str, Any]) -> str:
        events = data.get("events") or []
        latest = events[-1] if events else None
        if not latest:
            return f"订单 {data.get('order_id')} 目前还没有物流轨迹，可能尚未揽收。"
        text = (
            f"订单 {data.get('order_id')} 最新物流：{latest.get('time', '')[:16]} "
            f"{latest.get('status', '')}，{latest.get('desc', '')}。"
        )
        if data.get("stale_hours", 0) >= 48:
            text += f"轨迹已停滞 {data['stale_hours']:.0f} 小时，属于物流异常，可申请 10 元延迟补偿。"
        return text

    def _answer_policy(self, data: dict[str, Any]) -> str:
        results = data.get("results") or []
        if not results:
            return "没有检索到对应规则，建议转人工核实。"
        top = results[0]
        body = top["content"].replace("\n", " ")[:120].rstrip("。")
        return f"根据【{top['policy_id']}】《{top['title']}》{top['section']}：{body}。如需办理，请提供订单号。"


def build_llm(settings: Settings):
    if settings.offline:
        return MockLLM(settings)
    return DeepSeekLLM(settings)

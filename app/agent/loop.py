"""Agent 主循环：Function Calling → 工具执行 → 事实沉淀 → 护栏校验 → 修复/转人工。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings, business_now
from ..llm import Usage, build_llm
from ..rag import PolicyStore
from ..store import OrderStore
from ..tools import ToolContext, build_registry
from ..trace import ProgressFn, Tracer, new_trace_id
from . import guardrails
from .guardrails import GuardContext
from .memory import Session, SessionStore
from .prompts import REPAIR_PROMPT, SUMMARIZE_PROMPT, build_system_prompt

MAX_TOOL_CONTENT = 8000


@dataclass
class ToolRun:
    name: str
    args: dict[str, Any]
    ok: bool
    result: dict[str, Any]
    latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "preview": guardrails.preview(self.result, 500),
            "policy_ids": self.result.get("policy_ids", []),
        }


@dataclass
class TurnResult:
    answer: str
    session_id: str
    trace_id: str
    tool_runs: list[ToolRun] = field(default_factory=list)
    violations: list[dict[str, Any]] = field(default_factory=list)
    repaired: bool = False
    handoff: bool = False
    blocked_input: bool = False
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    usage_estimated: bool = False
    latency_ms: float = 0.0
    cost: float = 0.0
    model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "tool_runs": [run.as_dict() for run in self.tool_runs],
            "violations": self.violations,
            "repaired": self.repaired,
            "handoff": self.handoff,
            "blocked_input": self.blocked_input,
            "metrics": {
                "steps": self.steps,
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
                "usage_estimated": self.usage_estimated,
                "cost_cny": self.cost,
                "latency_ms": self.latency_ms,
                "model": self.model,
            },
        }


class CareAgent:
    def __init__(
        self,
        settings: Settings,
        llm=None,
        orders: OrderStore | None = None,
        policies: PolicyStore | None = None,
        sessions: SessionStore | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm or build_llm(settings)
        self.orders = orders or OrderStore.from_data_dir(settings.data_dir)
        self.policies = policies or PolicyStore.load(settings.data_dir / "policies")
        self.registry = build_registry()
        self.sessions = sessions or SessionStore(settings.runtime_dir / "sessions.json")

    # ------------------------------------------------------------------ 主流程
    def run(
        self,
        session: Session,
        user_text: str,
        emit: ProgressFn | None = None,
        tracer: Tracer | None = None,
    ) -> TurnResult:
        started = time.perf_counter()
        trace_id = tracer.trace_id if tracer else new_trace_id()
        tracer = tracer or Tracer(self.settings.trace_dir, trace_id)

        session.turn += 1
        ctx = ToolContext(
            now=business_now(),
            orders=self.orders,
            policies=self.policies,
            session_id=session.id,
            turn=session.turn,
            meta={"auto_limit": self.settings.auto_approve_limit},
        )

        usage = Usage()
        tool_runs: list[ToolRun] = []
        violations: list[dict[str, Any]] = []
        repaired = False
        handoff = False
        steps = 0
        usage_estimated = False
        draft_len = [0]

        tracer.log(
            "turn_start",
            session_id=session.id,
            user_id=session.user_id,
            turn=session.turn,
            user_text=user_text,
            mode=getattr(self.llm, "name", "unknown"),
        )
        self._emit(emit, "turn_start", {"session_id": session.id, "turn": session.turn})

        def finish(answer: str, blocked_input: bool = False) -> TurnResult:
            session.messages.append({"role": "user", "content": user_text})
            session.messages.append({"role": "assistant", "content": answer})
            session.last_trace_id = trace_id
            cost = usage.cost(getattr(self.llm, "name", "deepseek"))
            latency = round((time.perf_counter() - started) * 1000, 1)
            self.sessions.save()
            tracer.log(
                "turn_end",
                answer=answer,
                steps=steps,
                handoff=handoff,
                repaired=repaired,
                blocked_input=blocked_input,
                violations=violations,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_cny=cost,
                latency_ms=latency,
            )
            result = TurnResult(
                answer=answer,
                session_id=session.id,
                trace_id=trace_id,
                tool_runs=tool_runs,
                violations=violations,
                repaired=repaired,
                handoff=handoff,
                blocked_input=blocked_input,
                steps=steps,
                usage=usage,
                usage_estimated=usage_estimated,
                latency_ms=latency,
                cost=cost,
                model=getattr(self.llm, "name", ""),
            )
            self._emit(emit, "final", result.as_dict())
            return result

        # ---------------------------------------------------------- 输入护栏
        input_violations = guardrails.check_input(user_text)
        if guardrails.blocking(input_violations):
            violations.extend(v.as_dict() for v in input_violations)
            tracer.log("input_blocked", violations=[v.as_dict() for v in input_violations])
            self._emit(
                emit,
                "guardrail",
                {"stage": "input", "action": "block", "violations": [v.as_dict() for v in input_violations]},
            )
            return finish(guardrails.SAFE_INPUT_REPLY, blocked_input=True)

        # ---------------------------------------------------------- 构建上下文
        user = self.orders.user(session.user_id)
        system_prompt = build_system_prompt(
            ctx.now, session.user_id, user.get("tier", "普通会员"), self._facts_text(session)
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if session.summary:
            messages.append({"role": "system", "content": "## 历史对话摘要\n" + session.summary})
        messages.extend(session.messages[-self.settings.max_history_messages :])
        messages.append({"role": "user", "content": user_text})

        authorized = self._recent_authorized(session)
        retrieved = self._recent_policies(session)
        requires_human = self._recent_requires_human(session)
        successful_tools = self._recent_successful(session)

        def on_token(token: str) -> None:
            draft_len[0] += len(token)
            self._emit(emit, "token", {"text": token})

        answer = ""
        token_cb = on_token if emit else None

        # ---------------------------------------------------------- 工具循环
        for step in range(1, self.settings.max_steps + 1):
            steps = step
            try:
                chat = self.llm.chat(messages, tools=self.registry.schemas(), on_token=token_cb)
            except Exception as exc:  # 网络/额度等异常降级为转人工
                tracer.log("llm_error", step=step, error=f"{type(exc).__name__}: {exc}")
                self._safe_handoff(ctx, session, tool_runs, tracer, emit)
                handoff = True
                answer = self._handoff_text(tool_runs)
                if emit:
                    self._emit(emit, "reset", {})
                return finish(answer)

            usage = usage + chat.usage
            usage_estimated = usage_estimated or chat.estimated_usage
            tracer.log(
                "llm_call",
                step=step,
                model=chat.model,
                latency_ms=chat.latency_ms,
                prompt_tokens=chat.usage.prompt_tokens,
                completion_tokens=chat.usage.completion_tokens,
                finish_reason=chat.finish_reason,
                estimated_usage=chat.estimated_usage,
            )
            message = chat.message
            messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                if draft_len[0] > 0:
                    self._emit(emit, "reset", {})
                    draft_len[0] = 0
                for call in tool_calls:
                    name = call.get("function", {}).get("name", "")
                    raw_args = call.get("function", {}).get("arguments", "{}")
                    try:
                        parsed_args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        parsed_args = {"_raw": raw_args}

                    self._emit(emit, "tool_start", {"name": name, "args": parsed_args})
                    result = self.registry.execute(name, raw_args, ctx)
                    run = ToolRun(
                        name=name,
                        args=parsed_args,
                        ok=bool(result.get("ok")),
                        result=result,
                        latency_ms=float(result.get("latency_ms", 0.0)),
                    )
                    tool_runs.append(run)
                    tracer.log(
                        "tool_call",
                        step=step,
                        name=name,
                        args=parsed_args,
                        ok=run.ok,
                        latency_ms=run.latency_ms,
                        result=result,
                    )
                    self._emit(
                        emit,
                        "tool_end",
                        {
                            "name": name,
                            "ok": run.ok,
                            "latency_ms": run.latency_ms,
                            "preview": guardrails.preview(result, 700),
                            "policy_ids": result.get("policy_ids", []),
                        },
                    )

                    content = json.dumps(result, ensure_ascii=False, default=str)
                    if len(content) > MAX_TOOL_CONTENT:
                        content = content[:MAX_TOOL_CONTENT] + "…(truncated)"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "content": content,
                        }
                    )
                    authorized, requires_human = self._absorb_facts(
                        session, name, result, authorized, requires_human, successful_tools
                    )
                    if name == "transfer_to_human" and result.get("ok"):
                        handoff = True
                    for pid in result.get("policy_ids") or []:
                        session.facts.setdefault("retrieved", {})[pid] = session.turn
                retrieved = self._recent_policies(session)
                continue

            answer = (message.get("content") or "").strip()
            if answer:
                break
            tracer.log("empty_answer", step=step)
            break

        if not answer:
            self._safe_handoff(ctx, session, tool_runs, tracer, emit)
            handoff = True
            answer = self._handoff_text(tool_runs)
            if emit:
                self._emit(emit, "reset", {})
            return finish(answer)

        # ---------------------------------------------------------- 输出护栏
        guard_ctx = GuardContext(
            authorized_max_refund=authorized,
            retrieved_policy_ids=set(retrieved),
            successful_tools=set(successful_tools),
            requires_human=requires_human,
            tool_calls=len(tool_runs),
        )
        output_violations = guardrails.check_output(answer, guard_ctx)
        violations.extend(v.as_dict() for v in output_violations)
        blocks = guardrails.blocking(output_violations)
        tracer.log(
            "guardrail_check",
            violations=[v.as_dict() for v in output_violations],
            blocking=[v.as_dict() for v in blocks],
        )

        if blocks:
            self._emit(
                emit,
                "guardrail",
                {
                    "stage": "output",
                    "action": "repair",
                    "violations": [v.as_dict() for v in blocks],
                },
            )
            repaired = True
            rewritten, repair_usage, error = self._repair(
                messages, answer, blocks, authorized, successful_tools, retrieved
            )
            usage = usage + repair_usage
            if error:
                tracer.log("repair_error", error=error)

            if rewritten:
                if emit:
                    self._emit(emit, "reset", {})
                answer = rewritten
                recheck = guardrails.check_output(answer, guard_ctx)
                violations.extend({**v.as_dict(), "after_repair": True} for v in recheck)
                tracer.log(
                    "guardrail_recheck",
                    violations=[v.as_dict() for v in recheck],
                    blocking=[v.as_dict() for v in guardrails.blocking(recheck)],
                )
                if guardrails.blocking(recheck):
                    self._safe_handoff(ctx, session, tool_runs, tracer, emit)
                    handoff = True
                    answer = self._handoff_text(tool_runs)
                    if emit:
                        self._emit(emit, "reset", {})
            else:
                self._safe_handoff(ctx, session, tool_runs, tracer, emit)
                handoff = True
                answer = self._handoff_text(tool_runs)
                if emit:
                    self._emit(emit, "reset", {})

        # ---------------------------------------------------------- 记忆整理
        summary_usage = self._maybe_summarize(session, tracer)
        usage = usage + summary_usage

        return finish(answer)

    # ---------------------------------------------------------------- 事实沉淀
    def _absorb_facts(
        self,
        session: Session,
        name: str,
        result: dict[str, Any],
        authorized: float | None,
        requires_human: bool,
        successful: set[str],
    ) -> tuple[float | None, bool]:
        data = result.get("data") or {}

        if name == "query_order" and result.get("ok"):
            # 记住用户正在咨询的订单，后续「退货」「查物流」等指代无需再问订单号
            session.facts["last_order"] = {
                "order_id": data.get("order_id", ""),
                "status": data.get("status"),
                "paid_amount": data.get("paid_amount", 0.0),
                "turn": session.turn,
            }

        if name == "check_after_sale_eligibility" and result.get("ok"):
            order_id = data.get("order_id", "")
            session.facts.setdefault("orders", {})[order_id] = {
                "eligible": data.get("eligible"),
                "decision": data.get("decision"),
                "max_refund": data.get("max_refund", 0.0),
                "requires_human": data.get("requires_human", False),
                "policy_ids": data.get("policy_ids", []),
                "request_type": data.get("request_type"),
                "reason": data.get("reason"),
                "turn": session.turn,
            }
            amount = float(data.get("max_refund") or 0.0)
            authorized = amount if authorized is None else max(authorized, amount)
            requires_human = requires_human or bool(data.get("requires_human"))

        if name == "create_after_sale_ticket" and result.get("ok"):
            successful.add(name)
            session.facts["last_ticket"] = {
                "ticket_id": data.get("ticket_id"),
                "status": data.get("status"),
                "order_id": data.get("order_id"),
                "turn": session.turn,
            }
            requires_human = requires_human or bool(data.get("requires_human"))
            amount = float(data.get("max_refund") or 0.0)
            authorized = amount if authorized is None else max(authorized, amount)

        if name == "transfer_to_human" and result.get("ok"):
            successful.add(name)
            session.facts["last_handoff"] = {
                "ticket_id": data.get("ticket_id"),
                "turn": session.turn,
            }

        return authorized, requires_human

    # ------------------------------------------------------------------ 修复
    def _repair(
        self,
        messages: list[dict[str, Any]],
        answer: str,
        blocks: list[guardrails.Violation],
        authorized: float | None,
        successful: set[str],
        retrieved: set[str],
    ) -> tuple[str, Usage, str]:
        repair_messages = [
            {
                "role": "system",
                "content": REPAIR_PROMPT.format(
                    violations=guardrails.format_violations(blocks),
                    authorized=f"¥{authorized:.0f}" if authorized is not None else "未授权任何金额",
                    tools="、".join(sorted(successful)) or "无",
                    policies="、".join(sorted(retrieved)) or "无",
                ),
            },
            {
                "role": "user",
                "content": "原始对话（最后一条是你需要重写的回复）：\n"
                + json.dumps(
                    [
                        {"role": m.get("role"), "content": m.get("content", "")[:500]}
                        for m in messages
                        if m.get("role") in ("user", "assistant") and m.get("content")
                    ],
                    ensure_ascii=False,
                )
                + "\n\n待重写的回复：\n"
                + answer,
            },
        ]
        try:
            chat = self.llm.chat(repair_messages, temperature=0.0)
            return (chat.message.get("content") or "").strip(), chat.usage, ""
        except Exception as exc:
            return "", Usage(), f"{type(exc).__name__}: {exc}"

    # -------------------------------------------------------------- 转人工兜底
    def _safe_handoff(
        self,
        ctx: ToolContext,
        session: Session,
        tool_runs: list[ToolRun],
        tracer: Tracer,
        emit: ProgressFn | None,
    ) -> None:
        existing = session.facts.get("last_handoff") or {}
        if existing.get("turn") == session.turn and existing.get("ticket_id"):
            return

        order_id = ""
        for run in reversed(tool_runs):
            data = run.result.get("data") or {}
            if data.get("order_id"):
                order_id = data["order_id"]
                break
        args = {
            "order_id": order_id,
            "reason": "智能客服校验未通过，需要人工介入",
            "summary": "自动兜底转接：模型回复多次未通过合规校验。",
        }
        self._emit(emit, "tool_start", {"name": "transfer_to_human", "args": args})
        result = self.registry.execute("transfer_to_human", args, ctx)
        latency = float(result.get("latency_ms", 0.0))
        tool_runs.append(
            ToolRun(name="transfer_to_human", args=args, ok=bool(result.get("ok")), result=result, latency_ms=latency)
        )
        session.facts["last_handoff"] = {
            "ticket_id": (result.get("data") or {}).get("ticket_id"),
            "turn": session.turn,
        }
        tracer.log("safe_handoff", result=result)
        self._emit(
            emit,
            "tool_end",
            {
                "name": "transfer_to_human",
                "ok": bool(result.get("ok")),
                "latency_ms": latency,
                "preview": guardrails.preview(result, 500),
            },
        )

    def _handoff_text(self, tool_runs: list[ToolRun]) -> str:
        ticket = ""
        for run in reversed(tool_runs):
            if run.name == "transfer_to_human" and run.ok:
                ticket = (run.result.get("data") or {}).get("ticket_id", "")
                break
        suffix = f"（工单 {ticket}）" if ticket else ""
        return (
            "抱歉，为了确保信息准确，我已为您转接人工客服"
            f"{suffix}，预计 10 分钟内响应。您不需要重复描述问题。"
        )

    # ------------------------------------------------------------------ 记忆
    @staticmethod
    def _facts_text(session: Session) -> str:
        lines: list[str] = []
        last_order = session.facts.get("last_order") or {}
        if last_order and session.turn - int(last_order.get("turn", -99)) <= 3:
            lines.append(
                f"- 用户当前咨询订单：{last_order.get('order_id')}（"
                f"状态={last_order.get('status')}，"
                f"实付 ¥{float(last_order.get('paid_amount') or 0):.0f}）；"
                "用户后续提到「退货 / 换货 / 物流」等，默认指该订单，不要重复索要订单号"
            )
        for order_id, fact in list(session.facts.get("orders", {}).items())[-4:]:
            if session.turn - int(fact.get("turn", 0)) > 3:
                continue
            lines.append(
                f"- 订单 {order_id}：类型={fact.get('request_type')}；"
                f"原因={fact.get('reason') or '未记录'}；"
                f"结论={fact.get('decision')}；"
                f"最多可退 ¥{float(fact.get('max_refund') or 0):.0f}；"
                f"需人工={fact.get('requires_human')}"
            )
        policies = [
            pid
            for pid, turn in session.facts.get("retrieved", {}).items()
            if session.turn - int(turn) <= 3
        ]
        if policies:
            lines.append("- 本会话已检索政策：" + "、".join(sorted(policies)))
        ticket = session.facts.get("last_ticket")
        if ticket and session.turn - int(ticket.get("turn", 0)) <= 3:
            lines.append(
                f"- 最近工单：{ticket.get('ticket_id')}（{ticket.get('status')}，订单 {ticket.get('order_id')}）"
            )
        return "\n".join(lines)

    @staticmethod
    def _recent_authorized(session: Session) -> float | None:
        values = [
            float(fact.get("max_refund") or 0.0)
            for fact in session.facts.get("orders", {}).values()
            if session.turn - int(fact.get("turn", 0)) <= 3
        ]
        return max(values) if values else None

    @staticmethod
    def _recent_policies(session: Session) -> set[str]:
        return {
            pid
            for pid, turn in session.facts.get("retrieved", {}).items()
            if session.turn - int(turn) <= 3
        }

    @staticmethod
    def _recent_requires_human(session: Session) -> bool:
        return any(
            fact.get("requires_human")
            for fact in session.facts.get("orders", {}).values()
            if session.turn - int(fact.get("turn", 0)) <= 3
        )

    @staticmethod
    def _recent_successful(session: Session) -> set[str]:
        successful: set[str] = set()
        if session.turn - int(session.facts.get("last_ticket", {}).get("turn", -99)) <= 3:
            successful.add("create_after_sale_ticket")
        if session.turn - int(session.facts.get("last_handoff", {}).get("turn", -99)) <= 3:
            successful.add("transfer_to_human")
        return successful

    def _maybe_summarize(self, session: Session, tracer: Tracer) -> Usage:
        if len(session.messages) <= 16:
            return Usage()
        old = session.messages[:-8]
        session.messages = session.messages[-8:]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        prompt = SUMMARIZE_PROMPT.format(summary=session.summary or "无", transcript=transcript)

        if getattr(self.llm, "name", "") == "mock":
            session.summary = transcript[-200:]
            tracer.log("memory_summarized", mode="heuristic", length=len(session.summary))
            return Usage()

        try:
            chat = self.llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.1, response_format=None
            )
            session.summary = (chat.message.get("content") or "").strip()[:600]
            tracer.log("memory_summarized", mode="llm", length=len(session.summary))
            return chat.usage
        except Exception as exc:
            tracer.log("memory_summarize_error", error=str(exc))
            session.summary = transcript[-200:]
            return Usage()

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _emit(emit: ProgressFn | None, event: str, data: dict[str, Any]) -> None:
        if emit is None:
            return
        try:
            emit(event, data)
        except Exception:
            pass

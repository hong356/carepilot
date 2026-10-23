"""离线评测执行器。

用法：
    python -m evals.run                 # 有 Key 时走 DeepSeek
    python -m evals.run --mock          # 离线 Mock，验证评测链路
    python -m evals.run --judge         # 额外调用 deepseek-reasoner 质检打分
    python -m evals.run --case E01,E16  # 只跑指定用例
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent import CareAgent  # noqa: E402
from app.agent.memory import SessionStore  # noqa: E402
from app.config import business_now, load_settings  # noqa: E402
from app.llm import DeepSeekLLM, MockLLM  # noqa: E402

from .judge import judge_case  # noqa: E402

CASES_PATH = Path(__file__).parent / "cases.jsonl"


@dataclass
class CaseResult:
    case_id: str
    title: str
    passed: bool
    checks: list[dict[str, Any]]
    answer: str
    tools: list[str]
    steps: int
    latency_ms: float
    tokens: int
    cost: float
    violations: list[dict[str, Any]]
    handoff: bool
    trace_id: str
    judge: dict[str, Any] | None = None
    judge_cost: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_cases() -> list[dict[str, Any]]:
    cases = []
    with CASES_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str = "") -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def _evaluate(case: dict[str, Any], run) -> list[dict[str, Any]]:
    expect = case.get("expect", {})
    checks: list[dict[str, Any]] = []

    tool_set = set(run["tools"])
    for name in expect.get("must_call", []):
        _check(checks, f"must_call:{name}", name in tool_set, f"实际调用 {sorted(tool_set)}")
    for name in expect.get("must_not_call", []):
        _check(checks, f"must_not_call:{name}", name not in tool_set, f"实际调用 {sorted(tool_set)}")

    eligibility = run.get("eligibility")
    if "eligible" in expect:
        ok = eligibility is not None and eligibility.get("eligible") is expect["eligible"]
        _check(
            checks,
            "eligible",
            ok,
            "缺少规则判定结果" if eligibility is None else f"期望 {expect['eligible']}，实际 {eligibility.get('eligible')}",
        )
    if "requires_human" in expect:
        ok = eligibility is not None and eligibility.get("requires_human") is expect["requires_human"]
        _check(
            checks,
            "requires_human",
            ok,
            "缺少规则判定结果" if eligibility is None else f"期望 {expect['requires_human']}，实际 {eligibility.get('requires_human')}",
        )
    if "max_refund_lte" in expect:
        ok = eligibility is not None and eligibility.get("max_refund", 1e9) <= expect["max_refund_lte"] + 0.01
        _check(
            checks,
            "max_refund_lte",
            ok,
            "缺少规则判定结果" if eligibility is None else f"上限 {expect['max_refund_lte']}，实际 {eligibility.get('max_refund')}",
        )
    if "max_refund_gte" in expect:
        ok = eligibility is not None and eligibility.get("max_refund", -1) >= expect["max_refund_gte"] - 0.01
        _check(
            checks,
            "max_refund_gte",
            ok,
            "缺少规则判定结果" if eligibility is None else f"下限 {expect['max_refund_gte']}，实际 {eligibility.get('max_refund')}",
        )

    ticket = run.get("ticket")
    if "ticket_status" in expect:
        ok = ticket is not None and ticket.get("status") == expect["ticket_status"]
        _check(
            checks,
            "ticket_status",
            ok,
            "未创建工单" if ticket is None else f"期望 {expect['ticket_status']}，实际 {ticket.get('status')}",
        )
    if "ticket_duplicated" in expect:
        ok = ticket is not None and bool(ticket.get("duplicated")) is expect["ticket_duplicated"]
        _check(
            checks,
            "ticket_duplicated",
            ok,
            "未创建工单" if ticket is None else f"duplicated={ticket.get('duplicated')}",
        )

    answer = run["answer"]
    for keyword in expect.get("include", []):
        _check(checks, f"include:{keyword}", keyword in answer, f"回复中未出现「{keyword}」")
    for keyword in expect.get("exclude", []):
        _check(checks, f"exclude:{keyword}", keyword not in answer, f"回复中出现禁用词「{keyword}」")
    for policy_id in expect.get("must_cite", []):
        _check(checks, f"cite:{policy_id}", policy_id in answer, "回复未引用该政策编号")
    if expect.get("must_cite_any"):
        ok = any(pid in answer for pid in expect["must_cite_any"])
        _check(checks, "cite_any", ok, f"未引用任一政策编号 {expect['must_cite_any']}")

    if "handoff" in expect:
        _check(checks, "handoff", run["handoff"] is expect["handoff"], f"handoff={run['handoff']}")
    if "blocked_input" in expect:
        _check(
            checks,
            "blocked_input",
            run["blocked_input"] is expect["blocked_input"],
            f"blocked_input={run['blocked_input']}",
        )
    if "max_block_violations" in expect:
        blocks = [v for v in run["violations"] if v.get("severity") == "block"]
        _check(
            checks,
            "max_block_violations",
            len(blocks) <= expect["max_block_violations"],
            f"阻断级违规 {len(blocks)} 条（上限 {expect['max_block_violations']}）",
        )
    return checks


def run_case(case: dict[str, Any], llm, settings) -> dict[str, Any]:
    agent = CareAgent(settings, llm=llm, sessions=SessionStore(None))
    session = agent.sessions.get_or_create(None, case.get("user_id", "U1001"))

    results = [agent.run(session, turn) for turn in case["turns"]]

    tools: list[str] = []
    violations: list[dict[str, Any]] = []
    steps = 0
    latency = 0.0
    tokens = 0
    cost = 0.0
    handoff = False
    blocked_input = False
    eligibility: dict[str, Any] | None = None
    ticket: dict[str, Any] | None = None

    for result in results:
        tools.extend(run.name for run in result.tool_runs)
        violations.extend(result.violations)
        steps += result.steps
        latency += result.latency_ms
        tokens += result.usage.total_tokens
        cost += result.cost
        handoff = handoff or result.handoff
        blocked_input = blocked_input or result.blocked_input
        for run in result.tool_runs:
            if not run.ok:
                continue
            if run.name == "check_after_sale_eligibility":
                eligibility = run.result.get("data") or eligibility
            if run.name == "create_after_sale_ticket":
                ticket = run.result.get("data") or ticket

    last = results[-1]
    payload = {
        "answer": last.answer,
        "tools": tools,
        "eligibility": eligibility,
        "ticket": ticket,
        "violations": violations,
        "handoff": handoff,
        "blocked_input": blocked_input,
    }
    checks = _evaluate(case, payload)
    return {
        "case": case,
        "answer": last.answer,
        "trace_id": last.trace_id,
        "tools": tools,
        "steps": steps,
        "latency_ms": latency,
        "tokens": tokens,
        "cost": cost,
        "violations": violations,
        "handoff": handoff,
        "checks": checks,
        "passed": all(item["ok"] for item in checks),
        "eligibility": eligibility,
        "ticket": ticket,
        "blocked_input": blocked_input,
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p
    lower, upper = int(k), min(int(k) + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (k - lower)


def summarize(results: list[CaseResult], cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    expected_tools = [
        tool
        for case in cases
        for tool in case.get("expect", {}).get("must_call", [])
    ]
    found_tools = sum(
        1
        for case, result in zip(cases, results)
        for tool in case.get("expect", {}).get("must_call", [])
        if tool in set(result.tools)
    )
    judge_scores = [
        r.judge.get("score")
        for r in results
        if r.judge and isinstance(r.judge.get("score"), (int, float))
    ]
    return {
        "cases": total,
        "passed": passed,
        "success_rate": round(passed / total, 4) if total else 0.0,
        "tool_recall": round(found_tools / len(expected_tools), 4) if expected_tools else 1.0,
        "avg_steps": round(sum(r.steps for r in results) / total, 2) if total else 0.0,
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / total, 1) if total else 0.0,
        "p50_latency_ms": round(percentile([r.latency_ms for r in results], 0.5), 1),
        "p95_latency_ms": round(percentile([r.latency_ms for r in results], 0.95), 1),
        "total_tokens": sum(r.tokens for r in results),
        "total_cost_cny": round(sum(r.cost for r in results), 6),
        "judge_cost_cny": round(sum(r.judge_cost for r in results), 6),
        "block_violations": sum(
            1 for r in results for v in r.violations if v.get("severity") == "block"
        ),
        "handoffs": sum(1 for r in results if r.handoff),
        "judge_avg": round(statistics.mean(judge_scores), 2) if judge_scores else None,
        "judge_count": len(judge_scores),
    }


def _md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _latency(ms: float) -> str:
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.2f}s"


def build_report(
    meta: dict[str, Any],
    summary: dict[str, Any],
    results: list[CaseResult],
) -> str:
    lines: list[str] = []
    lines.append("# CarePilot 评测报告\n")
    lines.append(f"- 生成时间：{meta['generated_at']}")
    lines.append(f"- 运行模式：{meta['mode']}（模型 `{meta['model']}`）")
    lines.append(f"- 业务时间：{meta['business_now']}")
    lines.append(f"- 用例数：{summary['cases']} · 通过：{summary['passed']}")
    if meta["mode"] == "offline-mock":
        lines.append(
            "\n> ⚠️ 本次为离线 Mock 运行：只验证评测链路、规则引擎与护栏是否工作，"
            "**不代表 DeepSeek 模型的真实能力**。配置 `DEEPSEEK_API_KEY` 后重新运行可得到真实指标。\n"
        )

    lines.append("\n## 总览\n")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 任务成功率 | {summary['success_rate'] * 100:.1f}% |")
    lines.append(f"| 工具调用召回率 | {summary['tool_recall'] * 100:.1f}% |")
    lines.append(f"| 平均步数 | {summary['avg_steps']} |")
    lines.append(f"| 平均延迟 | {_latency(summary['avg_latency_ms'])} |")
    lines.append(f"| P50 / P95 延迟 | {_latency(summary['p50_latency_ms'])} / {_latency(summary['p95_latency_ms'])} |")
    lines.append(f"| 总 tokens | {summary['total_tokens']:,} |")
    lines.append(f"| 总成本（对话） | ¥{summary['total_cost_cny']:.4f} |")
    if summary["judge_count"]:
        lines.append(f"| LLM 裁判均分 | {summary['judge_avg']} / 5（{summary['judge_count']} 条） |")
        lines.append(f"| 总成本（含裁判） | ¥{summary['total_cost_cny'] + summary['judge_cost_cny']:.4f} |")
    lines.append(f"| 阻断级护栏触发 | {summary['block_violations']} |")
    lines.append(f"| 转人工 | {summary['handoffs']} |")

    lines.append("\n## 用例明细\n")
    header = "| 用例 | 结果 | 失败项 | 工具调用 | 步数 | 延迟 | tokens | 成本 |"
    lines.append(header)
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for result in results:
        failed = [c["name"] for c in result.checks if not c["ok"]]
        tools = ", ".join(dict.fromkeys(result.tools)) or "—"
        lines.append(
            f"| {result.case_id} {_md(result.title)} | {'✅' if result.passed else '❌'} | "
            f"{_md(', '.join(failed)) or '—'} | {_md(tools)} | {result.steps} | "
            f"{_latency(result.latency_ms)} | {result.tokens:,} | ¥{result.cost:.4f} |"
        )

    failures = [r for r in results if not r.passed]
    if failures:
        lines.append("\n## 失败详情\n")
        for result in failures:
            lines.append(f"### {result.case_id} {result.title}\n")
            for check in result.checks:
                if not check["ok"]:
                    lines.append(f"- ❌ `{check['name']}`：{_md(check['detail'])}")
            lines.append(f"\n回复：{_md(result.answer[:500])}\n")

    lines.append("\n## 评测口径\n")
    lines.append("- **任务成功率**：单条用例的全部断言通过，才算成功。")
    lines.append("- **工具召回率**：期望调用的工具中，实际被调用的比例。")
    lines.append("- **阻断级护栏触发**：输出/输入护栏判定为 block 的次数，越低越好。")
    lines.append("- **金额断言**：以规则引擎返回的 `max_refund` 为准，不比对自然语言表述。")
    lines.append("- 业务时间固定为 `DEMO_NOW`，保证时效类用例可复现。")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="CarePilot 离线评测")
    parser.add_argument("--mock", action="store_true", help="强制离线 Mock")
    parser.add_argument("--judge", action="store_true", help="启用 deepseek-reasoner 质检")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    parser.add_argument("--case", default="", help="只跑指定用例，逗号分隔，如 E01,E16")
    parser.add_argument("--out", default="", help="报告输出路径（.md）")
    args = parser.parse_args()

    base = load_settings()
    settings = replace(base, trace_dir=ROOT / "runtime" / "eval_traces")
    if args.mock:
        settings = replace(settings, offline=True, api_key="")
    llm = MockLLM(settings) if settings.offline else DeepSeekLLM(settings)
    mode = "offline-mock" if settings.offline else "deepseek"

    cases = load_cases()
    if args.case:
        wanted = {item.strip().upper() for item in args.case.split(",") if item.strip()}
        cases = [c for c in cases if c["id"].upper() in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("没有匹配的用例。")
        return 1

    if args.judge and settings.offline:
        print("⚠️  离线模式不执行 LLM 裁判，已跳过 --judge。")
        args.judge = False

    print(f"运行模式：{mode} · 模型：{getattr(llm, 'name', '?')} · 用例：{len(cases)}")
    results: list[CaseResult] = []
    judge_usage_total = 0.0

    for index, case in enumerate(cases, 1):
        raw = run_case(case, llm, settings)
        judge_data = None
        judge_cost = 0.0
        if args.judge:
            judge_data, usage, judge_model = judge_case(settings, case, raw["answer"])
            judge_cost = usage.cost(judge_model)
            judge_usage_total += judge_cost

        result = CaseResult(
            case_id=case["id"],
            title=case["title"],
            passed=raw["passed"],
            checks=raw["checks"],
            answer=raw["answer"],
            tools=raw["tools"],
            steps=raw["steps"],
            latency_ms=raw["latency_ms"],
            tokens=raw["tokens"],
            cost=raw["cost"],
            violations=raw["violations"],
            handoff=raw["handoff"],
            trace_id=raw["trace_id"],
            judge=judge_data,
            judge_cost=judge_cost,
        )
        results.append(result)
        mark = "PASS" if result.passed else "FAIL"
        extra = ""
        if judge_data:
            extra = f" judge={judge_data.get('score', '?')}"
        print(f"[{index:02d}/{len(cases)}] {case['id']} {case['title']}  {mark}{extra}")

    summary = summarize(results, cases)
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        "generated_at": generated_at,
        "mode": mode,
        "model": "mock" if settings.offline else settings.chat_model,
        "judge_model": None if not args.judge else settings.judge_model,
        "business_now": business_now().isoformat(),
        "case_file": str(CASES_PATH),
    }
    report = build_report(meta, summary, results)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else ROOT / "reports" / f"eval_{stamp}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "meta": meta,
                "summary": summary,
                "cases": [result.as_dict() for result in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\n成功率 {summary['success_rate'] * 100:.1f}% ({summary['passed']}/{summary['cases']}) · "
        f"工具召回 {summary['tool_recall'] * 100:.1f}% · "
        f"P95 {_latency(summary['p95_latency_ms'])} · "
        f"tokens {summary['total_tokens']:,} · ¥{summary['total_cost_cny']:.4f}"
    )
    if summary["judge_count"]:
        print(f"裁判均分 {summary['judge_avg']} / 5（{summary['judge_count']} 条）")
    print(f"报告：{out_path}")
    print(f"数据：{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

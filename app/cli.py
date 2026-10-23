"""命令行演示：python -m app.cli"""
from __future__ import annotations

import argparse
import json
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def _print_orders(agent, user_id: str) -> None:
    table = Table(title=f"演示订单 · {user_id}", show_lines=False, header_style="bold cyan")
    table.add_column("订单号", style="bold")
    table.add_column("状态")
    table.add_column("商品")
    table.add_column("实付", justify="right")
    for order in agent.orders.orders_of_user(user_id):
        table.add_row(
            order["order_id"],
            order["status"],
            order["items"][0]["title"],
            f"¥{order['paid_amount']:.0f}",
        )
    console.print(table)


def _run_once(agent, session, text: str, verbose: bool = False) -> None:
    tool_count = 0

    def emit(event: str, data: dict) -> None:
        nonlocal tool_count
        if event == "tool_start":
            tool_count += 1
            if verbose:
                args = json.dumps(data.get("args", {}), ensure_ascii=False)
                console.print(f"\n  [dim cyan]↳ {data['name']}[/dim cyan] [dim]{args}[/dim]")
        elif event == "tool_end":
            if verbose:
                mark = "[green]✓[/green]" if data.get("ok") else "[red]✗[/red]"
                console.print(f"  [dim]{mark} {data['name']} · {data.get('latency_ms', 0)}ms[/dim]")
        elif event == "token":
            console.print(data["text"], end="", markup=False, highlight=False, soft_wrap=True)
        elif event == "guardrail":
            codes = "、".join(v["code"] for v in data.get("violations", []))
            console.print(f"\n  [yellow]⚠ 护栏触发：{codes}（{data.get('action')}）[/yellow]")

    console.print(Panel(Text(text, style="bold"), title="你", border_style="cyan", expand=False))
    console.print("\n[bold green]CarePilot[/bold green] ", end="")
    try:
        result = agent.run(session, text, emit=emit)
    except Exception as exc:
        console.print(f"\n[red]请求失败：{type(exc).__name__}: {exc}[/red]")
        return
    console.print()
    console.print(
        f"[dim]{result.steps} 步 · {result.usage.total_tokens} tokens · "
        f"¥{result.cost:.4f} · {result.latency_ms:.0f}ms · trace {result.trace_id}[/dim]"
    )
    if result.handoff:
        console.print("[yellow]→ 已转人工[/yellow]")
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="CarePilot 售后智能体 CLI")
    parser.add_argument("--user", default="U1001", choices=["U1001", "U1002", "U1003"])
    parser.add_argument("--offline", action="store_true", help="强制离线 Mock 模式")
    parser.add_argument("--once", metavar="TEXT", help="只跑一轮，便于脚本化演示")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示工具调用过程（默认只显示回答）")
    args = parser.parse_args()

    if args.offline:
        os.environ["CAREPILOT_OFFLINE"] = "1"

    from .agent import CareAgent
    from .config import load_settings

    settings = load_settings()
    agent = CareAgent(settings)
    session = agent.sessions.get_or_create(None, args.user)

    console.print(
        Panel.fit(
            "[bold]CarePilot · 电商售后智能体[/bold]\n"
            f"模式：[cyan]{settings.mode}[/cyan]  "
            f"用户：[cyan]{args.user}[/cyan]\n"
            "命令：/new 新建会话  /orders 查看订单  /trace 最近轨迹  /quit 退出",
            border_style="green",
        )
    )
    _print_orders(agent, args.user)

    if args.once:
        _run_once(agent, session, args.once, args.verbose)
        return

    while True:
        try:
            text = console.input("\n[bold cyan]你 ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见。")
            break
        if not text:
            continue
        if text in {"/quit", "/exit", "quit", "exit"}:
            console.print("再见。")
            break
        if text == "/new":
            session = agent.sessions.get_or_create(None, args.user)
            console.print("[dim]已新建会话。[/dim]")
            continue
        if text == "/orders":
            _print_orders(agent, args.user)
            continue
        if text == "/trace":
            if session.last_trace_id:
                path = settings.trace_dir / f"{session.last_trace_id}.jsonl"
                console.print(f"[dim]{path}[/dim]")
            else:
                console.print("[dim]当前会话还没有轨迹。[/dim]")
            continue
        if text == "/help":
            console.print("[dim]/new 新建会话 · /orders 查看订单 · /trace 轨迹路径 · /quit 退出[/dim]")
            continue

        _run_once(agent, session, text, args.verbose)


if __name__ == "__main__":
    sys.exit(main())

"""
Sanity check: what days/symbols are available on each venue,
how many trades & L2 messages are in each.

Run:
    uv run python scripts/01_inspect.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loaders import hl, dydx  # noqa: E402

load_dotenv()
console = Console()

SYMBOLS = [s.strip() for s in os.getenv("LAB_SYMBOLS", "BTC,ETH,SOL").split(",")]


def main():
    console.rule("[bold cyan]HL — available days per symbol (trades)[/bold cyan]")
    t = Table()
    t.add_column("symbol")
    t.add_column("trades_days")
    t.add_column("l2_days")
    for sym in SYMBOLS:
        td = hl.list_days("trades", sym)
        l2 = hl.list_days("l2_book", sym)
        t.add_row(sym,
                  f"{len(td)}: {td[0]}…{td[-1]}" if td else "0",
                  f"{len(l2)}: {l2[0]}…{l2[-1]}" if l2 else "0")
    console.print(t)

    console.rule("[bold cyan]dYdX — available days[/bold cyan]")
    dd = dydx.list_days()
    console.print(f"{len(dd)} days: {dd[0]} … {dd[-1]}" if dd else "no days")

    common_days = sorted(set(hl.list_days("trades", SYMBOLS[0])) & set(dd))
    console.rule(f"[bold green]Common days for cross-venue analysis: {len(common_days)}")
    for d in common_days:
        console.print(f"  {d}")

    if not common_days:
        console.print("[red]No overlap between venues. Wait for accumulation, then re-run.[/red]")
        return

    sample_day = common_days[-1]
    console.rule(f"[bold cyan]Sample row counts for {sample_day}[/bold cyan]")
    t2 = Table()
    t2.add_column("symbol")
    t2.add_column("HL trades")
    t2.add_column("HL L2 top changes")
    t2.add_column("dYdX trades")
    t2.add_column("dYdX L2 top changes")
    for sym in SYMBOLS:
        try:
            ht = hl.read_trades_day(sym, sample_day)
            n_ht = ht.height
        except Exception as e:
            n_ht = f"err: {e}"
        try:
            console.print(f"[dim]parsing HL L2 {sym} {sample_day}…[/dim]")
            hl2 = hl.l2_top_of_book_day(sym, sample_day)
            n_hl2 = hl2.height
        except Exception as e:
            n_hl2 = f"err: {e}"
        try:
            dt_ = dydx.read_trades_day(sym, sample_day)
            n_dt = dt_.height
        except Exception as e:
            n_dt = f"err: {e}"
        try:
            console.print(f"[dim]parsing dYdX L2 {sym} {sample_day}…[/dim]")
            dl2 = dydx.l2_top_of_book_day(sym, sample_day)
            n_dl2 = dl2.height
        except Exception as e:
            n_dl2 = f"err: {e}"
        t2.add_row(sym, str(n_ht), str(n_hl2), str(n_dt), str(n_dl2))
    console.print(t2)


if __name__ == "__main__":
    main()

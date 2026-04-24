"""
Honest backtest of basis mean-reversion across HL ↔ dYdX.

Strategy:
  - Compute rolling mean & std of basis_bps over WINDOW seconds (default 1h).
  - When |basis - rolling_mean| > Z * rolling_std → enter contra trade:
      * if basis << mean (too negative) → buy basis: long HL, short dYdX
      * if basis >> mean (too positive) → sell basis: short HL, long dYdX
  - Exit when basis crosses rolling_mean back, OR after MAX_HOLD seconds.
  - PnL per trade in bps = (entry_basis - exit_basis) × direction
        ── direction = +1 for "sell basis", -1 for "buy basis"
  - Round-trip cost subtracted (FEE_BPS).

What we report (per symbol):
  - n_trades, win_rate
  - gross / net PnL (bps total + per-trade mean)
  - average hold time
  - daily implied PnL on $X notional

Run:
    uv run python scripts/05_backtest_mr.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()
console = Console()

SYMBOLS = [s.strip() for s in os.getenv("LAB_SYMBOLS", "BTC,ETH,SOL").split(",")]
OUT = Path(os.getenv("LAB_OUT", "./out")).resolve()

# Strategy params (defaults — overridable via env)
WINDOW_SEC = int(os.getenv("BT_WINDOW_SEC", "3600"))      # rolling stats window
ENTRY_Z = float(os.getenv("BT_ENTRY_Z", "2.0"))           # enter at ±Zσ
MAX_HOLD_SEC = int(os.getenv("BT_MAX_HOLD", "600"))       # 10 min max hold
FEE_BPS = float(os.getenv("BT_FEE_BPS", "10.0"))          # round-trip ALL costs
NOTIONAL_USD = float(os.getenv("BT_NOTIONAL_USD", "1000"))


def backtest_symbol(sym: str) -> dict:
    src = OUT / "mids_all" / f"{sym}.parquet"
    if not src.exists():
        return {"symbol": sym, "error": "no mids file"}

    df = pl.read_parquet(src).sort("ts_sec")
    if df.height < WINDOW_SEC + 100:
        return {"symbol": sym, "error": f"too few rows ({df.height} < {WINDOW_SEC+100})"}

    # rolling stats: shift by 1 to use only past info (no lookahead)
    df = df.with_columns(
        pl.col("basis_bps").rolling_mean(window_size=WINDOW_SEC, min_samples=WINDOW_SEC // 4).shift(1).alias("rm"),
        pl.col("basis_bps").rolling_std(window_size=WINDOW_SEC, min_samples=WINDOW_SEC // 4).shift(1).alias("rstd"),
    ).drop_nulls(["rm", "rstd"]).filter(pl.col("rstd") > 0)

    if df.is_empty():
        return {"symbol": sym, "error": "no rows after rolling stats"}

    ts = df["ts_sec"].to_numpy()
    b = df["basis_bps"].to_numpy()
    rm = df["rm"].to_numpy()
    rstd = df["rstd"].to_numpy()
    z = (b - rm) / rstd

    trades = []
    i = 0
    n = len(b)
    while i < n:
        if abs(z[i]) >= ENTRY_Z:
            entry_idx = i
            entry_b = b[i]
            entry_rm = rm[i]
            # direction: if basis is too positive, we expect it to FALL → sell basis (direction = +1)
            # if basis is too negative, we expect it to RISE → buy basis (direction = -1)
            direction = 1 if z[i] > 0 else -1

            # walk forward to exit
            exit_idx = i
            while exit_idx < n - 1:
                exit_idx += 1
                # exit on cross of rolling mean
                if direction == 1 and b[exit_idx] <= entry_rm:
                    break
                if direction == -1 and b[exit_idx] >= entry_rm:
                    break
                # timeout
                if (ts[exit_idx] - ts[entry_idx]) >= MAX_HOLD_SEC:
                    break

            exit_b = b[exit_idx]
            hold_s = int(ts[exit_idx] - ts[entry_idx])
            gross_bps = (entry_b - exit_b) * direction  # PnL in basis-bps space
            net_bps = gross_bps - FEE_BPS

            trades.append({
                "t_entry": int(ts[entry_idx]),
                "t_exit": int(ts[exit_idx]),
                "hold_s": hold_s,
                "entry_b": float(entry_b),
                "exit_b": float(exit_b),
                "entry_z": float(z[entry_idx]),
                "direction": direction,
                "gross_bps": float(gross_bps),
                "net_bps": float(net_bps),
            })

            # cooldown — next bar after exit
            i = exit_idx + 1
        else:
            i += 1

    if not trades:
        return {"symbol": sym, "n_trades": 0, "note": "no trades triggered"}

    tdf = pl.DataFrame(trades)
    n_t = tdf.height
    wins = (tdf["net_bps"] > 0).sum()
    gross_total = float(tdf["gross_bps"].sum())
    net_total = float(tdf["net_bps"].sum())
    avg_hold = float(tdf["hold_s"].mean())
    span_hours = (df["ts_sec"][-1] - df["ts_sec"][0]) / 3600.0

    pnl_usd_total = net_total * NOTIONAL_USD / 10_000  # bps × notional / 10000
    pnl_usd_per_hour = pnl_usd_total / span_hours if span_hours > 0 else 0.0
    pnl_usd_per_day = pnl_usd_per_hour * 24

    return {
        "symbol": sym,
        "rows": df.height,
        "span_h": round(span_hours, 2),
        "n_trades": n_t,
        "win_rate": round(wins / n_t, 3),
        "gross_total_bps": round(gross_total, 1),
        "net_total_bps": round(net_total, 1),
        "net_per_trade_bps": round(net_total / n_t, 2),
        "avg_hold_s": round(avg_hold, 1),
        "pnl_usd_total": round(pnl_usd_total, 2),
        "pnl_usd_per_day": round(pnl_usd_per_day, 2),
        "trades_df": tdf,
    }


def main():
    console.rule(f"[bold cyan]Mean-reversion backtest")
    console.print(
        f"window={WINDOW_SEC}s  entry_z={ENTRY_Z}  max_hold={MAX_HOLD_SEC}s  "
        f"fee={FEE_BPS}bps  notional=${NOTIONAL_USD:.0f}"
    )

    t = Table()
    t.add_column("symbol")
    t.add_column("rows")
    t.add_column("span_h")
    t.add_column("trades")
    t.add_column("win_rate")
    t.add_column("gross_total_bps")
    t.add_column("net_total_bps")
    t.add_column("net/trade")
    t.add_column("avg_hold_s")
    t.add_column("PnL $")
    t.add_column("$/day implied")

    for sym in SYMBOLS:
        r = backtest_symbol(sym)
        if "error" in r:
            console.print(f"[yellow]{sym}: {r['error']}[/yellow]")
            continue
        if r.get("n_trades", 0) == 0:
            console.print(f"[yellow]{sym}: no trades triggered[/yellow]")
            continue
        t.add_row(
            r["symbol"], str(r["rows"]), str(r["span_h"]), str(r["n_trades"]),
            f"{r['win_rate']:.0%}",
            str(r["gross_total_bps"]), str(r["net_total_bps"]),
            str(r["net_per_trade_bps"]), str(r["avg_hold_s"]),
            f"${r['pnl_usd_total']}", f"${r['pnl_usd_per_day']}",
        )
        # save trades csv per symbol
        trades_path = OUT / "backtest" / f"{sym}_trades.csv"
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        r["trades_df"].write_csv(trades_path)

    console.print(t)
    console.print(
        f"[dim]gross_total_bps = sum of (entry_basis - exit_basis) × direction across trades.\n"
        f"net = gross - {FEE_BPS} bps round-trip per trade.\n"
        f"PnL $ scales with notional. $/day implied = naive extrapolation from {{}} day window.[/dim]"
    )


if __name__ == "__main__":
    main()

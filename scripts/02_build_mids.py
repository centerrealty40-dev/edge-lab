"""
Build per-second mid-price tables for both venues, save as parquet.

For each (symbol, day):
    out/mids/<symbol>/<ymd>.parquet
        columns: ts_sec (i64 unix seconds, UTC),
                 hl_mid (f64), dydx_mid (f64),
                 hl_spread_bps (f64), dydx_spread_bps (f64),
                 basis_bps (f64)

Run:
    uv run python scripts/02_build_mids.py
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loaders import hl, dydx  # noqa: E402

load_dotenv()
console = Console()

SYMBOLS = [s.strip() for s in os.getenv("LAB_SYMBOLS", "BTC,ETH,SOL").split(",")]
OUT = Path(os.getenv("LAB_OUT", "./out")).resolve()


def to_per_second_mid(top_df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """top_df has columns ts_ms, bid, ask, bid_sz, ask_sz.
    Returns one row per (ts_sec) with last observation in that second."""
    if top_df.is_empty():
        return pl.DataFrame()
    df = (
        top_df
        .with_columns(
            ((pl.col("bid") + pl.col("ask")) / 2.0).alias(f"{prefix}_mid"),
            ((pl.col("ask") - pl.col("bid")) / ((pl.col("bid") + pl.col("ask")) / 2.0) * 10_000).alias(f"{prefix}_spread_bps"),
            (pl.col("ts_ms") // 1000).alias("ts_sec"),
        )
        .group_by("ts_sec")
        .agg(
            pl.col(f"{prefix}_mid").last(),
            pl.col(f"{prefix}_spread_bps").last(),
        )
        .sort("ts_sec")
    )
    return df


def build_for_day(symbol: str, d: date) -> pl.DataFrame | None:
    console.print(f"[cyan]→ {symbol} {d}: HL L2…[/cyan]")
    hl_top = hl.l2_top_of_book_day(symbol, d)
    if hl_top.is_empty():
        console.print(f"[yellow]  no HL L2 data for {symbol} {d}[/yellow]")
        return None
    hl_sec = to_per_second_mid(hl_top, "hl")
    console.print(f"   HL: {hl_top.height} top changes → {hl_sec.height} seconds")

    console.print(f"[cyan]→ {symbol} {d}: dYdX L2 (delta replay)…[/cyan]")
    dx_top = dydx.l2_top_of_book_day(symbol, d)
    if dx_top.is_empty():
        console.print(f"[yellow]  no dYdX L2 data for {symbol} {d}[/yellow]")
        return None
    dx_sec = to_per_second_mid(dx_top, "dydx")
    console.print(f"   dYdX: {dx_top.height} top changes → {dx_sec.height} seconds")

    joined = hl_sec.join(dx_sec, on="ts_sec", how="inner")
    if joined.is_empty():
        console.print("[yellow]  no overlapping seconds[/yellow]")
        return None

    joined = joined.with_columns(
        ((pl.col("hl_mid") - pl.col("dydx_mid")) / pl.col("dydx_mid") * 10_000).alias("basis_bps"),
    ).sort("ts_sec")
    return joined


def main():
    common = sorted(set(hl.list_days("l2_book", SYMBOLS[0])) & set(dydx.list_days()))
    if not common:
        console.print("[red]No overlapping days. Aborting.[/red]")
        return

    for sym in SYMBOLS:
        for d in common:
            out = OUT / "mids" / sym / f"{d.strftime('%Y%m%d')}.parquet"
            if out.exists():
                console.print(f"[dim]cached {sym} {d}[/dim]")
                continue
            df = build_for_day(sym, d)
            if df is None or df.is_empty():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out, compression="zstd")
            console.print(f"[green]wrote {out} ({df.height} rows)[/green]")


if __name__ == "__main__":
    main()

"""
Stats on the basis (HL_mid - dYdX_mid) per symbol.

Answers:
  - distribution of basis_bps
  - mean / std / quantiles
  - autocorrelation at 1s, 5s, 30s — does it mean-revert?
  - max excursions and durations

Run:
    uv run python scripts/03_basis_stats.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()
console = Console()

SYMBOLS = [s.strip() for s in os.getenv("LAB_SYMBOLS", "BTC,ETH,SOL").split(",")]
OUT = Path(os.getenv("LAB_OUT", "./out")).resolve()


def autocorr(series: pl.Series, lag: int) -> float:
    if series.len() < lag + 5:
        return float("nan")
    s = series.drop_nulls()
    if s.len() < lag + 5:
        return float("nan")
    a = s[:-lag]
    b = s[lag:]
    if a.len() != b.len() or a.std() == 0 or b.std() == 0:
        return float("nan")
    cov = ((a - a.mean()) * (b - b.mean())).mean()
    return float(cov / (a.std() * b.std()))


def main():
    for sym in SYMBOLS:
        sym_dir = OUT / "mids" / sym
        if not sym_dir.exists():
            console.print(f"[yellow]no mids for {sym}, run 02_build_mids first[/yellow]")
            continue
        files = sorted(sym_dir.glob("*.parquet"))
        if not files:
            continue

        df = pl.concat([pl.read_parquet(f) for f in files]).sort("ts_sec")
        b = df["basis_bps"]

        console.rule(f"[bold cyan]{sym} — basis_bps over {df.height} seconds")
        t = Table()
        t.add_column("metric")
        t.add_column("value")
        t.add_row("count", f"{b.len()}")
        t.add_row("mean", f"{b.mean():.3f}")
        t.add_row("std",  f"{b.std():.3f}")
        for q in (0.01, 0.05, 0.50, 0.95, 0.99):
            t.add_row(f"q{int(q*100):02d}", f"{b.quantile(q):.3f}")
        t.add_row("min",  f"{b.min():.3f}")
        t.add_row("max",  f"{b.max():.3f}")
        for lag in (1, 5, 30, 60, 300):
            t.add_row(f"autocorr lag={lag}s", f"{autocorr(b, lag):.4f}")
        # how often basis exceeds 5/10/20 bps
        for thr in (5, 10, 20, 50):
            cnt = (b.abs() > thr).sum()
            t.add_row(f"|basis|>{thr}bps", f"{cnt} ({cnt/b.len()*100:.2f}%)")
        console.print(t)

        # save the concatenated frame for the next stage
        out_concat = OUT / "mids_all" / f"{sym}.parquet"
        out_concat.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_concat, compression="zstd")
        console.print(f"[green]wrote concatenated {out_concat}[/green]")


if __name__ == "__main__":
    main()

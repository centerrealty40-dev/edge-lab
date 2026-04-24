"""
Visualize basis_bps over time for each symbol.
Saves PNG plots to out/plots/<symbol>.png — open them in a file browser to eyeball.

What to look for:
  - Does basis OSCILLATE around its mean (mean-reversion → tradable),
    or does it DRIFT and stay drifted (structural, not tradable)?
  - Do extreme excursions (|basis| > 2σ) come back to zero, or do they
    persist for hours?
  - Are there regime changes (sudden jumps to a new mean level)?

Run:
    uv run python scripts/04_plot_basis.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless server
import matplotlib.pyplot as plt
import polars as pl
from dotenv import load_dotenv
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()
console = Console()

SYMBOLS = [s.strip() for s in os.getenv("LAB_SYMBOLS", "BTC,ETH,SOL").split(",")]
OUT = Path(os.getenv("LAB_OUT", "./out")).resolve()


def plot_symbol(symbol: str) -> None:
    src = OUT / "mids_all" / f"{symbol}.parquet"
    if not src.exists():
        console.print(f"[yellow]no {src} (run 03_basis_stats first)[/yellow]")
        return

    df = pl.read_parquet(src).sort("ts_sec")
    n = df.height
    if n < 100:
        console.print(f"[yellow]{symbol}: only {n} rows, skipping plot[/yellow]")
        return

    ts = df["ts_sec"].to_numpy()
    basis = df["basis_bps"].to_numpy()
    mean = float(df["basis_bps"].mean())
    std = float(df["basis_bps"].std())

    # convert ts_sec to UTC datetime axis labels
    import datetime as dt
    times = [dt.datetime.utcfromtimestamp(int(t)) for t in ts]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(times, basis, lw=0.4, color="#2563eb", alpha=0.85)
    ax.axhline(mean, color="black", lw=0.8, ls="--", label=f"mean = {mean:.1f} bps")
    ax.axhline(mean + 2 * std, color="red",   lw=0.6, ls=":", alpha=0.7, label=f"±2σ ({2*std:.0f} bps)")
    ax.axhline(mean - 2 * std, color="red",   lw=0.6, ls=":", alpha=0.7)
    ax.axhline(0, color="gray", lw=0.4, alpha=0.4)
    ax.set_ylabel("basis_bps  (HL_mid - dYdX_mid) / dYdX_mid · 1e4")
    ax.set_title(f"{symbol}  basis over time   N={n} seconds   mean={mean:.1f}  std={std:.1f}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # spread chart below
    ax2 = axes[1]
    if "hl_spread_bps" in df.columns and "dydx_spread_bps" in df.columns:
        ax2.plot(times, df["hl_spread_bps"].to_numpy(),   lw=0.4, color="#16a34a", label="HL spread", alpha=0.7)
        ax2.plot(times, df["dydx_spread_bps"].to_numpy(), lw=0.4, color="#ea580c", label="dYdX spread", alpha=0.7)
        ax2.set_ylabel("spread_bps")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = OUT / "plots" / f"{symbol}_basis.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    console.print(f"[green]wrote {out_path}[/green]")

    # also a histogram
    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.hist(basis, bins=80, color="#2563eb", alpha=0.85)
    ax.axvline(mean, color="black", ls="--", label=f"mean={mean:.1f}")
    ax.axvline(mean + 2 * std, color="red", ls=":", alpha=0.7, label=f"±2σ")
    ax.axvline(mean - 2 * std, color="red", ls=":", alpha=0.7)
    ax.set_xlabel("basis_bps")
    ax.set_ylabel("count")
    ax.set_title(f"{symbol}  basis distribution   N={n}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    out_hist = OUT / "plots" / f"{symbol}_hist.png"
    fig2.savefig(out_hist, dpi=110)
    plt.close(fig2)
    console.print(f"[green]wrote {out_hist}[/green]")


def main():
    for sym in SYMBOLS:
        plot_symbol(sym)


if __name__ == "__main__":
    main()

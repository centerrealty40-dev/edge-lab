"""
Lead-lag analysis: who moves first, HL or dYdX?

Computes cross-correlation between log-returns of HL_mid and dYdX_mid
at lags from -MAX_LAG to +MAX_LAG seconds.

Interpretation:
  - max corr at positive lag k → HL leads dYdX by k seconds
    (HL changes now → dYdX moves k seconds later) → predictive trade on dYdX
  - max corr at negative lag k → dYdX leads HL
  - max corr at lag 0 → simultaneous, no lead-lag edge

Saves a plot to out/plots/<symbol>_xcorr.png.

Run:
    uv run python scripts/06_lead_lag.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
MAX_LAG = int(os.getenv("LL_MAX_LAG", "30"))


def cross_corr(r_hl: np.ndarray, r_dx: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (lags array, correlations array). Lag k > 0 means r_dx is shifted
    LEFT by k → we correlate r_hl(t) with r_dx(t+k), i.e. dYdX FOLLOWS HL by k sec.
    """
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.zeros(len(lags), dtype=np.float64)
    n = len(r_hl)
    for i, k in enumerate(lags):
        if k >= 0:
            a = r_hl[: n - k] if k > 0 else r_hl
            b = r_dx[k:] if k > 0 else r_dx
        else:
            a = r_hl[-k:]
            b = r_dx[: n + k]
        if len(a) < 100:
            corrs[i] = np.nan
            continue
        sa, sb = a.std(), b.std()
        if sa == 0 or sb == 0:
            corrs[i] = np.nan
        else:
            corrs[i] = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    return lags, corrs


def analyze(symbol: str) -> dict:
    src = OUT / "mids_all" / f"{symbol}.parquet"
    if not src.exists():
        return {"symbol": symbol, "error": "no mids file"}

    df = pl.read_parquet(src).sort("ts_sec")
    if df.height < 1000:
        return {"symbol": symbol, "error": f"only {df.height} rows"}

    # Ensure dense 1Hz grid by ffill (pads gaps where one venue was silent)
    full = pl.DataFrame({
        "ts_sec": pl.arange(int(df["ts_sec"][0]), int(df["ts_sec"][-1]) + 1, eager=True),
    })
    df = full.join(df, on="ts_sec", how="left").sort("ts_sec").fill_null(strategy="forward").drop_nulls()

    hl = df["hl_mid"].to_numpy()
    dx = df["dydx_mid"].to_numpy()

    # log returns 1s
    r_hl = np.diff(np.log(hl))
    r_dx = np.diff(np.log(dx))

    lags, corrs = cross_corr(r_hl, r_dx, MAX_LAG)
    best_idx = int(np.nanargmax(corrs))
    best_lag = int(lags[best_idx])
    best_corr = float(corrs[best_idx])

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(lags, corrs, width=0.8,
           color=["#16a34a" if k == best_lag else "#2563eb" for k in lags])
    ax.axvline(0, color="black", lw=0.6)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlabel("lag (sec)  positive ⇒ dYdX follows HL")
    ax.set_ylabel("correlation of 1s log-returns")
    ax.set_title(f"{symbol}  cross-correlation HL→dYdX   peak@lag={best_lag}s   ρ={best_corr:.3f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = OUT / "plots" / f"{symbol}_xcorr.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    return {
        "symbol": symbol,
        "rows": len(r_hl),
        "best_lag_s": best_lag,
        "best_corr": round(best_corr, 4),
        "corr_at_lag_0": round(float(corrs[MAX_LAG]), 4),
        "corr_at_lag_+1": round(float(corrs[MAX_LAG + 1]), 4),
        "corr_at_lag_-1": round(float(corrs[MAX_LAG - 1]), 4),
        "interpretation": (
            "HL leads dYdX" if best_lag > 0 else
            "dYdX leads HL" if best_lag < 0 else
            "no lead-lag (simultaneous)"
        ),
    }


def main():
    console.rule("[bold cyan]Lead-lag cross-correlation HL ↔ dYdX")
    t = Table()
    t.add_column("symbol")
    t.add_column("rows")
    t.add_column("ρ@lag=-1")
    t.add_column("ρ@lag=0")
    t.add_column("ρ@lag=+1")
    t.add_column("best lag")
    t.add_column("best ρ")
    t.add_column("interpretation")

    for sym in SYMBOLS:
        r = analyze(sym)
        if "error" in r:
            console.print(f"[yellow]{sym}: {r['error']}[/yellow]")
            continue
        t.add_row(
            r["symbol"], str(r["rows"]),
            str(r["corr_at_lag_-1"]), str(r["corr_at_lag_0"]), str(r["corr_at_lag_+1"]),
            f"{r['best_lag_s']:+d}s", str(r["best_corr"]),
            r["interpretation"],
        )

    console.print(t)
    console.print(
        "[dim]positive lag = dYdX follows HL with that delay → if peak at +k seconds,\n"
        "we can trade dYdX based on HL movements with k-second window.[/dim]"
    )


if __name__ == "__main__":
    main()

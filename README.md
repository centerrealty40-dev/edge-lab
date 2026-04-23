# edge-lab

Cross-venue research lab over **Hyperliquid** and **dYdX v4** live data
collected by `hl-research` / `hyperliquid-edge` and `dydx-edge`.

The first concrete question we want to answer:

> Is there a stable basis between HL and dYdX on BTC/ETH/SOL? How big, how
> often does it exceed transaction costs, and does it mean-revert?

If yes → we have a candidate for a market-neutral cross-venue strategy.
If no → we move on to the next hypothesis (lead/lag, flow imbalance, etc.).

## What's inside

```
edge-lab/
├── pyproject.toml
├── .env.example
├── README.md
├── loaders/
│   ├── hl.py        Hyperliquid:  daily JSONL trades + L2 snapshots
│   └── dydx.py      dYdX v4:      gzip JSONL trades + L2 deltas (replay required)
└── scripts/
    ├── 01_inspect.py        what data exists on each venue, how much
    ├── 02_build_mids.py     1-second mid prices joined across venues → parquet
    └── 03_basis_stats.py    distribution, autocorrelation, threshold hits
```

## Setup (server, KVM Ubuntu)

```bash
ssh root@srv1481733
cd /opt
git clone https://github.com/centerrealty40-dev/edge-lab.git
cd edge-lab
/root/.local/bin/uv sync
cp .env.example .env

/root/.local/bin/uv run python scripts/01_inspect.py
/root/.local/bin/uv run python scripts/02_build_mids.py
/root/.local/bin/uv run python scripts/03_basis_stats.py
```

`02_build_mids.py` is the slow one — it reconstructs the dYdX L2 book from
deltas. Expect ~minutes per (symbol, day) on a small KVM. Files cached
under `out/mids/<symbol>/<ymd>.parquet`, so re-runs are instant.

## What to look at

After `03_basis_stats.py`, scan the printed table:

- **`mean`, `std`** of `basis_bps`: is one venue systematically cheaper?
- **`autocorr lag=Xs`**: positive = persistent basis; negative = mean-reverts;
  near zero = noise (no edge).
- **`|basis|>10bps`** count: how often the spread exceeds round-trip fees
  (HL maker rebates + dYdX taker ≈ 5–10 bps total).

If autocorr is meaningfully negative at 30–300s and threshold hits are
non-trivial, we have something to backtest.

## Coexistence

Reads only — does not touch other projects' data folders. Lives at
`/opt/edge-lab/`. No systemd service (this is a research project, not a
streamer).

"""
Hyperliquid loader: reads daily JSONL from /opt/hl-research/data/stream/.

L2 format (snapshot per push):
    {"coin":"BTC","time":<ms>,"levels":[[bids],[asks]]}
    where each level = {"px":"<str>","sz":"<str>","n":<int>}

Trades:
    {"coin":"BTC","side":"B"|"A","px":"<str>","sz":"<str>",
     "time":<ms>,"hash":..., "tid":..., "users":[buyer, seller]}

Files are plain .jsonl per UTC day (some old files .jsonl.gz, we handle both).
"""

from __future__ import annotations

import gzip
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import polars as pl


def _data_root() -> Path:
    return Path(os.getenv("HL_DATA_ROOT", "/opt/hl-research/data")).resolve()


def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def _day_files(kind: str, coin: str, d: date) -> list[Path]:
    """kind ∈ {'trades', 'l2_book'}. Returns existing files for that day."""
    root = _data_root() / "stream" / kind / coin
    ymd = d.strftime("%Y-%m-%d")
    candidates = [root / f"{ymd}.jsonl", root / f"{ymd}.jsonl.gz"]
    return [p for p in candidates if p.exists() and p.stat().st_size > 0]


def list_coins(kind: str = "trades") -> list[str]:
    root = _data_root() / "stream" / kind
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def list_days(kind: str, coin: str) -> list[date]:
    root = _data_root() / "stream" / kind / coin
    if not root.exists():
        return []
    days: set[date] = set()
    for p in root.iterdir():
        # accept both .jsonl and .jsonl.gz
        name = p.name
        for suffix in (".jsonl", ".jsonl.gz"):
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                try:
                    days.add(datetime.strptime(stem, "%Y-%m-%d").date())
                except ValueError:
                    pass
                break
    return sorted(days)


def read_trades_day(coin: str, d: date) -> pl.DataFrame:
    """Returns DataFrame with columns: ts_ms (i64), side (str B/A), px (f64),
    sz (f64), tid (i64), hash (str). Empty DF if no file."""
    paths = _day_files("trades", coin, d)
    if not paths:
        return pl.DataFrame()
    # take the first (prefer .jsonl over .gz if both exist; both should be same content)
    df = pl.read_ndjson(paths[0])
    if df.is_empty():
        return df
    out = df.select(
        pl.col("time").cast(pl.Int64).alias("ts_ms"),
        pl.col("side").cast(pl.Utf8),
        pl.col("px").cast(pl.Float64),
        pl.col("sz").cast(pl.Float64),
        pl.col("tid").cast(pl.Int64) if "tid" in df.columns else pl.lit(None).alias("tid"),
        pl.col("hash").cast(pl.Utf8) if "hash" in df.columns else pl.lit(None).alias("hash"),
    )
    return out


def iter_l2_day(coin: str, d: date) -> Iterator[dict]:
    """Yields raw L2 messages so callers can do what they want without paying
    DataFrame allocation cost for 100M+ levels in memory."""
    import json
    for path in _day_files("l2_book", coin, d):
        with _open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def l2_top_of_book_day(coin: str, d: date) -> pl.DataFrame:
    """Reduce L2 snapshots to top-of-book series.
    Returns columns: ts_ms (i64), bid (f64), ask (f64), bid_sz (f64), ask_sz (f64).
    """
    rows: list[dict] = []
    for msg in iter_l2_day(coin, d):
        levels = msg.get("levels")
        if not isinstance(levels, list) or len(levels) != 2:
            continue
        bids, asks = levels[0], levels[1]
        if not bids or not asks:
            continue
        try:
            top_bid = bids[0]
            top_ask = asks[0]
            rows.append({
                "ts_ms": int(msg["time"]),
                "bid": float(top_bid["px"]),
                "ask": float(top_ask["px"]),
                "bid_sz": float(top_bid["sz"]),
                "ask_sz": float(top_ask["sz"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def date_range(start: date, end: date) -> Iterator[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def parse_window() -> tuple[date | None, date | None]:
    s = os.getenv("LAB_START_DATE") or None
    e = os.getenv("LAB_END_DATE") or None
    sd = datetime.strptime(s, "%Y-%m-%d").date() if s else None
    ed = datetime.strptime(e, "%Y-%m-%d").date() if e else None
    return sd, ed

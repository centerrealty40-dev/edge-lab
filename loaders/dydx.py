"""
dYdX v4 loader: reads /opt/dydx-edge/data/stream/<YYYYMMDD>/<MARKET>__<channel>.jsonl.gz.

WS message shapes:
  - "type":"subscribed", "channel":"v4_orderbook", "id":"BTC-USD",
        "contents":{"bids":[["75292","0.123"], ...], "asks":[["75440","0.05"], ...]}
        → initial full snapshot

  - "type":"channel_batch_data", "channel":"v4_orderbook", "id":"BTC-USD",
        "contents":[{"bids":[["75292","0"]]}, {"asks":[["75440","0.053"]]}, ...]
        → incremental deltas. size="0" = remove level.

  - "type":"channel_data", "channel":"v4_trades", "id":"BTC-USD",
        "contents":{"trades":[{...trade...}]}
        → trades

For mid-price reconstruction we need to:
  1. find the initial snapshot
  2. apply each delta sequentially to maintain bid/ask sorted books
  3. emit (ts, top_bid, top_ask) on every change (or sample to 1Hz)

We don't have a server-side timestamp on every message, but we have OS arrival
time when we wrote the line. dYdX doesn't include per-event ts in batch deltas,
so we attach OUR receive-time (file write order) as a monotonic proxy.
For 1Hz mid resampling that's accurate enough.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from sortedcontainers import SortedDict
from typing import Iterator

import polars as pl


SYMBOL_MAP = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}


def _data_root() -> Path:
    return Path(os.getenv("DYDX_DATA_ROOT", "/opt/dydx-edge/data")).resolve()


def hl_to_dydx(symbol: str) -> str:
    return SYMBOL_MAP.get(symbol, symbol)


def list_days() -> list[date]:
    root = _data_root() / "stream"
    if not root.exists():
        return []
    days: set[date] = set()
    for p in root.iterdir():
        if p.is_dir() and len(p.name) == 8 and p.name.isdigit():
            try:
                days.add(datetime.strptime(p.name, "%Y%m%d").date())
            except ValueError:
                pass
    return sorted(days)


def _file(d: date, market: str, channel: str) -> Path | None:
    p = _data_root() / "stream" / d.strftime("%Y%m%d") / f"{market}__{channel}.jsonl.gz"
    return p if p.exists() and p.stat().st_size > 0 else None


def iter_jsonl_gz(path: Path) -> Iterator[tuple[int | None, dict]]:
    """Read jsonl.gz robustly, yielding (recv_ms_or_None, msg).

    The streamer (post-fix) wraps each line as {"recv_ms": <int>, "msg": {...}}.
    For old (pre-fix) data we yield (None, raw_msg) so callers can fall back
    to the broken interpolation logic.

    Uses system gunzip via subprocess: tolerant to in-progress files lacking
    the end-of-stream marker.
    """
    proc = subprocess.Popen(
        ["gunzip", "-c", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=1 << 16,
    )
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and "recv_ms" in obj and "msg" in obj:
                yield (int(obj["recv_ms"]), obj["msg"])
            else:
                yield (None, obj)
    finally:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass


def read_trades_day(symbol: str, d: date) -> pl.DataFrame:
    """Returns: ts_ms (i64), side (str BUY/SELL), px (f64), sz (f64), tid (str).
    Empty DF if no file."""
    market = hl_to_dydx(symbol)
    p = _file(d, market, "v4_trades")
    if p is None:
        return pl.DataFrame()

    rows: list[dict] = []
    for _recv_ms, msg in iter_jsonl_gz(p):
        contents = msg.get("contents") or {}
        trades = contents.get("trades") if isinstance(contents, dict) else None
        if not isinstance(trades, list):
            continue
        for t in trades:
            try:
                # dYdX createdAt is ISO-8601 with Z
                ts_iso = t["createdAt"]
                ts_ms = int(datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp() * 1000)
                rows.append({
                    "ts_ms": ts_ms,
                    "side": t["side"],
                    "px": float(t["price"]),
                    "sz": float(t["size"]),
                    "tid": t.get("id", ""),
                })
            except (KeyError, ValueError, TypeError):
                continue
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def l2_top_of_book_day(symbol: str, d: date) -> pl.DataFrame:
    """Reconstructs L2 books from initial snapshot + deltas, emits top-of-book
    samples whenever the best bid or ask changes.
    Returns: ts_ms (i64), bid (f64), ask (f64), bid_sz (f64), ask_sz (f64).

    Time source: prefers `recv_ms` from the streamer wrapper. If absent (old
    pre-fix files), refuses to fabricate timestamps and returns an empty frame
    — interpolation across uneven message rates is misleading for analysis.
    """
    market = hl_to_dydx(symbol)
    p = _file(d, market, "v4_orderbook")
    if p is None:
        return pl.DataFrame()

    bids: SortedDict = SortedDict()
    asks: SortedDict = SortedDict()

    rows: list[tuple[int, float, float, float, float]] = []
    last_top: tuple[float | None, float | None, float | None, float | None] = (None, None, None, None)
    saw_any_recv_ms = False
    last_recv_ms: int | None = None

    for recv_ms, msg in iter_jsonl_gz(p):
        if recv_ms is not None:
            saw_any_recv_ms = True
            last_recv_ms = recv_ms
        t = msg.get("type")
        contents = msg.get("contents")

        if t == "subscribed" and isinstance(contents, dict):
            bids.clear()
            asks.clear()
            for px_s, sz_s in contents.get("bids") or []:
                sz = float(sz_s)
                if sz > 0:
                    bids[float(px_s)] = sz
            for px_s, sz_s in contents.get("asks") or []:
                sz = float(sz_s)
                if sz > 0:
                    asks[float(px_s)] = sz

        elif t == "channel_batch_data" and isinstance(contents, list):
            for delta in contents:
                if not isinstance(delta, dict):
                    continue
                for side in ("bids", "asks"):
                    if side in delta:
                        book = bids if side == "bids" else asks
                        for px_s, sz_s in delta[side]:
                            px = float(px_s)
                            sz = float(sz_s)
                            if sz == 0:
                                book.pop(px, None)
                            else:
                                book[px] = sz

        elif t == "channel_data" and isinstance(contents, dict):
            if "bids" in contents or "asks" in contents:
                for px_s, sz_s in contents.get("bids") or []:
                    sz = float(sz_s)
                    if sz > 0:
                        bids[float(px_s)] = sz
                    else:
                        bids.pop(float(px_s), None)
                for px_s, sz_s in contents.get("asks") or []:
                    sz = float(sz_s)
                    if sz > 0:
                        asks[float(px_s)] = sz
                    else:
                        asks.pop(float(px_s), None)

        if bids and asks and last_recv_ms is not None:
            top_bid_px = bids.keys()[-1]
            top_ask_px = asks.keys()[0]
            top = (top_bid_px, top_ask_px, bids[top_bid_px], asks[top_ask_px])
            if top != last_top:
                rows.append((last_recv_ms, top_bid_px, top_ask_px, bids[top_bid_px], asks[top_ask_px]))
                last_top = top

    if not saw_any_recv_ms:
        # Old data without recv_ms wrapping — don't fake timestamps.
        return pl.DataFrame()

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(
        rows,
        schema=[("ts_ms", pl.Int64), ("bid", pl.Float64), ("ask", pl.Float64),
                ("bid_sz", pl.Float64), ("ask_sz", pl.Float64)],
        orient="row",
    )


def date_range(start: date, end: date) -> Iterator[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

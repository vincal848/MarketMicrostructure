"""Databento MBP-10 (top-10 price levels) -> six-type flow, by snapshot diffs.

MBP records carry no order ids, and some book changes appear only in the
attached snapshot (verified on XNAS.ITCH SPY, 2025-11-11):

- An order replace is published as just the add at its new price; the
  removal at the old price shows up only in the snapshot.
- Adds and cancels sometimes have side 'N'.
- The level decrease caused by a fill can arrive on the next record, or on a
  later trade record of the same event, rather than as a cancel record.

So the action column is not trusted for passive flow. Every record's
snapshot is diffed against the previous one:

- A level *increase* is an add, typed LA / LI / LD against the previous
  best on its side.
- A level *decrease* first absorbs pending fill volume at that side and
  price (registered by trade records in the same event, i.e. the same
  `ts_event`); any remainder is a cancel.

Trades ('T') are market orders: side B is a buy aggressor and A a sell. Side
'N' is signed by price against the previous mid. Trades with the same
timestamp and aggressor merge, as in `flow.py`.

Only prices visible in *both* snapshots are compared. When a side shows all
ten levels, a price below the tenth bid (or above the tenth ask) may simply
have scrolled in or out of view.

Timestamps become nanoseconds since midnight New York time, the same clock
as ITCH. Prices become 1/10000 dollar integers, the ITCH/LOBSTER unit.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import pandas as pd

from microstructure.book import Side
from microstructure.flow import ClassifiedFlow, FlowRecorder, FlowType, add_type

FilePath = str | os.PathLike[str]

LEVELS = 10
PRICE_SCALE = 10_000
F_SNAPSHOT = 32
_BID_PX = [f"bid_px_{i:02d}" for i in range(LEVELS)]
_ASK_PX = [f"ask_px_{i:02d}" for i in range(LEVELS)]
_BID_SZ = [f"bid_sz_{i:02d}" for i in range(LEVELS)]
_ASK_SZ = [f"ask_sz_{i:02d}" for i in range(LEVELS)]
_COLUMNS = ["ts_event", "action", "side", "price", "size", "flags", *_BID_PX, *_ASK_PX, *_BID_SZ, *_ASK_SZ]

Snapshot = dict[int, int]  # price -> size, one side


def _ns_since_midnight_new_york(ts_utc_ns: pd.Series) -> np.ndarray:
    local = pd.to_datetime(ts_utc_ns, utc=True).dt.tz_convert("America/New_York")
    return (local - local.dt.normalize()).to_numpy().astype("timedelta64[ns]").astype(np.int64)


def _to_units(prices: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(prices), 0, np.rint(prices * PRICE_SCALE)).astype(np.int64)


class _MbpClassifier:
    """Streaming state: previous snapshot, pending fills, merged trades."""

    def __init__(self, tick: int) -> None:
        if tick <= 0:
            raise ValueError(f"tick must be positive, got {tick!r}")
        self.tick = tick
        self.recorder = FlowRecorder()
        self.book: dict[Side, Snapshot] = {Side.BID: {}, Side.ASK: {}}
        self.pending_fills: defaultdict[tuple[Side, int], int] = defaultdict(int)
        self.event_ts: int | None = None

    def feed(self, records: pd.DataFrame) -> None:
        ts = _ns_since_midnight_new_york(records["ts_event"])
        prices = _to_units(records["price"].to_numpy(dtype=np.float64))
        bid_px = _to_units(records[_BID_PX].to_numpy(dtype=np.float64))
        ask_px = _to_units(records[_ASK_PX].to_numpy(dtype=np.float64))
        bid_sz = records[_BID_SZ].to_numpy(dtype=np.int64)
        ask_sz = records[_ASK_SZ].to_numpy(dtype=np.int64)
        actions = records["action"].to_numpy()
        sides = records["side"].to_numpy()
        sizes = records["size"].to_numpy(dtype=np.int64)
        flags = records["flags"].to_numpy(dtype=np.int64)
        event_ts = records["ts_event"].to_numpy(dtype=np.int64)

        for row in range(len(records)):
            if event_ts[row] != self.event_ts:
                self.pending_fills.clear()
                self.event_ts = int(event_ts[row])
            new_book = {
                Side.BID: {int(p): int(q) for p, q in zip(bid_px[row], bid_sz[row], strict=True) if q > 0},
                Side.ASK: {int(p): int(q) for p, q in zip(ask_px[row], ask_sz[row], strict=True) if q > 0},
            }
            if flags[row] & F_SNAPSHOT or actions[row] == "R":
                self.book = new_book  # book (re)initialisation, not order flow
                continue
            if actions[row] == "T":
                self._trade(int(ts[row]), str(sides[row]), int(prices[row]), int(sizes[row]))
            self._diff(int(ts[row]), new_book)
            self.book = new_book

    def _trade(self, ts: int, side: str, price: int, qty: int) -> None:
        if side == "N":
            bids, asks = self.book[Side.BID], self.book[Side.ASK]
            if not bids or not asks or 2 * price == max(bids) + min(asks):
                self.recorder.unsigned_trades += 1
                return
            side = "B" if 2 * price > max(bids) + min(asks) else "A"
        aggressor_buys = side == "B"
        self.pending_fills[(Side.ASK if aggressor_buys else Side.BID, price)] += qty
        self.recorder.market(ts, FlowType.MB if aggressor_buys else FlowType.MS, qty)

    def _diff(self, ts: int, new_book: dict[Side, Snapshot]) -> None:
        for side in (Side.BID, Side.ASK):
            old, new = self.book[side], new_book[side]
            best_before = (max(old) if side is Side.BID else min(old)) if old else None
            visible = self._comparable(side, old, new)
            changes = sorted(
                (
                    (price, new.get(price, 0) - old.get(price, 0))
                    for price in old.keys() | new.keys()
                    if visible(price)
                ),
                key=lambda change: change[1] > 0,  # decreases (cancel half of a replace) before increases
            )
            for price, delta in changes:
                if delta < 0:
                    self._decrease(ts, side, price, -delta, best_before)
                elif delta > 0:
                    kind, distance = add_type(side, price, best_before, self.tick)
                    self.recorder.passive(ts, kind, delta, distance)

    def _decrease(self, ts: int, side: Side, price: int, qty: int, best_before: int | None) -> None:
        filled = min(qty, self.pending_fills.get((side, price), 0))
        if filled:
            self.pending_fills[(side, price)] -= filled
        cancelled = qty - filled
        if cancelled > 0:
            depth = 0 if best_before is None else round(abs(best_before - price) / self.tick)
            self.recorder.passive(ts, FlowType.C, cancelled, depth)

    @staticmethod
    def _comparable(side: Side, old: Snapshot, new: Snapshot) -> Callable[[int], bool]:
        """Predicate for prices visible in both snapshots."""

        def edge(levels: Snapshot) -> float:
            if len(levels) < LEVELS:
                return -math.inf if side is Side.BID else math.inf
            return float(min(levels) if side is Side.BID else max(levels))

        if side is Side.BID:
            floor = max(edge(old), edge(new))
            return lambda price: price >= floor
        ceiling = min(edge(old), edge(new))
        return lambda price: price <= ceiling


def classify_mbp10(records: pd.DataFrame, tick: int) -> ClassifiedFlow:
    """Classify a DataFrame of MBP-10 records (Databento CSV columns)."""
    classifier = _MbpClassifier(tick)
    classifier.feed(records[_COLUMNS])
    return classifier.recorder.finish()


def read_mbp10(path: FilePath, tick: int, chunk_rows: int = 1_000_000) -> ClassifiedFlow:
    """Stream a Databento MBP-10 CSV file in chunks and classify it.

    State carries across chunks, so chunk boundaries never split an event.
    """
    classifier = _MbpClassifier(tick)
    for chunk in pd.read_csv(path, usecols=_COLUMNS, chunksize=chunk_rows):
        classifier.feed(chunk)
    return classifier.recorder.finish()

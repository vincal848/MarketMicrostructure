"""Classify order events into the six-type flow alphabet the Hawkes model uses.

    MB  market buy                LA  limit add at the best
    MS  market sell               LI  limit add inside the spread
    C   cancel (full or partial)  LD  limit add away from the best

Each event is judged against the book *before* it (docs/ARCHITECTURE.md has
the full mapping table). Classification replays the events through a
`Replayer`, so the audit counts from Phase 1 come along with the flow.

Conventions:
- One aggressive order often executes against several resting orders,
  producing several executions with the same timestamp. Consecutive
  executions with the same timestamp and aggressor side are merged into one
  market order.
- A replace is a cancel followed by an add. The add is classified against
  the book after the cancel, since that is the book it arrives in.
- Hidden executions have no displayed side, so the aggressor is signed by
  price against mid (above mid: buy). Trades exactly at mid, or with a
  one-sided book, are counted in `unsigned_trades` and dropped.
- Cross executions (ITCH 'C' with printable = N) are not market orders.
- `distance` is in ticks: improvement for LI, depth behind the best for LD
  and C, zero otherwise.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from microstructure.book import OrderBook, Side
from microstructure.events import (
    AddOrder,
    CancelOrder,
    DeleteOrder,
    ExecuteOrder,
    ExecuteOrderWithPrice,
    HiddenTrade,
    OrderEvent,
    ReplaceOrder,
)
from microstructure.hawkes import EventStream
from microstructure.replay import Replayer, ReplayReport

IntArray = npt.NDArray[np.int64]
NS_PER_SECOND = 1_000_000_000


class FlowType(IntEnum):
    MB = 0
    MS = 1
    LA = 2
    LI = 3
    LD = 4
    C = 5


N_FLOW_TYPES = len(FlowType)


@dataclass(frozen=True)
class ClassifiedFlow:
    """Classified events as parallel arrays, in time order."""

    ts: IntArray
    kind: IntArray
    qty: IntArray
    distance: IntArray
    unsigned_trades: int = 0
    replay: ReplayReport = field(default_factory=ReplayReport)

    def __post_init__(self) -> None:
        arrays = {
            name: np.asarray(getattr(self, name), dtype=np.int64)
            for name in ("ts", "kind", "qty", "distance")
        }
        lengths = {a.shape for a in arrays.values()}
        if len(lengths) != 1 or arrays["ts"].ndim != 1:
            raise ValueError("ts, kind, qty and distance must be 1-D arrays of equal length")
        if np.any(np.diff(arrays["ts"]) < 0):
            raise ValueError("ts must be non-decreasing")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    def __len__(self) -> int:
        return int(self.ts.shape[0])

    def between(self, start_ns: int, end_ns: int) -> ClassifiedFlow:
        """Events with start_ns <= ts < end_ns."""
        keep = (self.ts >= start_ns) & (self.ts < end_ns)
        return ClassifiedFlow(self.ts[keep], self.kind[keep], self.qty[keep], self.distance[keep])

    def to_stream(self, start_ns: int, end_ns: int) -> EventStream:
        """The window as a Hawkes `EventStream`, in seconds from `start_ns`."""
        window = self.between(start_ns, end_ns)
        times = (window.ts - start_ns) / NS_PER_SECOND
        return EventStream(times, window.kind, horizon=(end_ns - start_ns) / NS_PER_SECOND)


@dataclass(frozen=True)
class FlowMarks:
    """Empirical mark distributions: order sizes per type, and tick distances
    for the types that have one (LI, LD, C)."""

    sizes: dict[FlowType, IntArray]
    distances: dict[FlowType, IntArray]

    def within(self, max_ticks: int) -> FlowMarks:
        """Drop distance samples beyond `max_ticks` from the touch.

        Real books carry stub quotes far from the market (SPY 2019-01-30: LD
        p90 = 795 ticks, max 2.6 million). They are irrelevant to the
        microstructure, but in a simulator they pile up as a far reservoir,
        and whenever the near book thins they become the best price, so the
        mid jumps by hundreds of ticks.
        """
        return FlowMarks(self.sizes, {kind: d[d <= max_ticks] for kind, d in self.distances.items()})


def marks(flow: ClassifiedFlow) -> FlowMarks:
    sizes = {kind: flow.qty[flow.kind == kind] for kind in FlowType}
    distances = {kind: flow.distance[flow.kind == kind] for kind in (FlowType.LI, FlowType.LD, FlowType.C)}
    return FlowMarks(sizes, distances)


def add_type(side: Side, price: int, same_side_best: int | None, tick: int) -> tuple[FlowType, int]:
    """LA / LI / LD and tick distance for an add at `price` on `side`."""
    if same_side_best is None or price == same_side_best:
        return FlowType.LA, 0
    improvement = price - same_side_best if side is Side.BID else same_side_best - price
    if improvement > 0:
        return FlowType.LI, round(improvement / tick)
    return FlowType.LD, round(-improvement / tick)


class FlowRecorder:
    """Accumulates classified events, merging same-time market orders."""

    def __init__(self) -> None:
        self._ts: list[int] = []
        self._kind: list[int] = []
        self._qty: list[int] = []
        self._distance: list[int] = []
        self._pending_market: list[int] | None = None  # [ts, kind, qty]
        self.unsigned_trades = 0

    def market(self, ts: int, kind: FlowType, qty: int) -> None:
        pending = self._pending_market
        if pending is not None and pending[0] == ts and pending[1] == kind:
            pending[2] += qty
            return
        self._flush()
        self._pending_market = [ts, int(kind), qty]

    def passive(self, ts: int, kind: FlowType, qty: int, distance: int) -> None:
        self._flush()
        self._append(ts, int(kind), qty, distance)

    def finish(self, replay: ReplayReport | None = None) -> ClassifiedFlow:
        self._flush()
        return ClassifiedFlow(
            np.asarray(self._ts, dtype=np.int64),
            np.asarray(self._kind, dtype=np.int64),
            np.asarray(self._qty, dtype=np.int64),
            np.asarray(self._distance, dtype=np.int64),
            unsigned_trades=self.unsigned_trades,
            replay=replay if replay is not None else ReplayReport(),
        )

    def _flush(self) -> None:
        if self._pending_market is not None:
            ts, kind, qty = self._pending_market
            self._append(ts, kind, qty, 0)
            self._pending_market = None

    def _append(self, ts: int, kind: int, qty: int, distance: int) -> None:
        self._ts.append(ts)
        self._kind.append(kind)
        self._qty.append(qty)
        self._distance.append(distance)


def _depth_ticks(book: OrderBook, side: Side, price: int, tick: int) -> int:
    best = book.best_bid() if side is Side.BID else book.best_ask()
    return 0 if best is None else round(abs(best - price) / tick)


def classify(events: Iterable[OrderEvent], tick: int) -> ClassifiedFlow:
    """Replay `events` and classify each into the six-type alphabet."""
    if tick <= 0:
        raise ValueError(f"tick must be positive, got {tick!r}")
    replayer = Replayer(audit_continuous_only=True)
    book = replayer.book
    recorder = FlowRecorder()

    def add(ts: int, side: Side, price: int, qty: int) -> None:
        best = book.best_bid() if side is Side.BID else book.best_ask()
        kind, distance = add_type(side, price, best, tick)
        recorder.passive(ts, kind, qty, distance)

    def cancel(ts: int, order_id: int, qty: int | None) -> None:
        order = book.resting_order(order_id)
        cancelled = order.qty if qty is None else min(qty, order.qty)
        recorder.passive(ts, FlowType.C, cancelled, _depth_ticks(book, order.side, order.price, tick))

    def execution(ts: int, order_id: int, qty: int) -> None:
        resting_side = book.resting_order(order_id).side
        recorder.market(ts, FlowType.MB if resting_side is Side.ASK else FlowType.MS, qty)

    for event in events:
        match event:
            case AddOrder():
                add(event.ts, event.side, event.price, event.qty)
            case CancelOrder() if event.order_id in book:
                cancel(event.ts, event.order_id, event.qty)
            case DeleteOrder() if event.order_id in book:
                cancel(event.ts, event.order_id, None)
            case ReplaceOrder() if event.order_id in book:
                side = book.resting_order(event.order_id).side
                cancel(event.ts, event.order_id, None)
                replayer.apply(DeleteOrder(ts=event.ts, order_id=event.order_id))
                add(event.ts, side, event.price, event.qty)
                event = AddOrder(
                    ts=event.ts, order_id=event.new_order_id, side=side, price=event.price, qty=event.qty
                )
            case ExecuteOrder() if event.order_id in book:
                execution(event.ts, event.order_id, event.qty)
            case ExecuteOrderWithPrice() if event.printable and event.order_id in book:
                execution(event.ts, event.order_id, event.qty)
            case HiddenTrade():
                bid, ask = book.best_bid(), book.best_ask()
                if bid is None or ask is None or 2 * event.price == bid + ask:
                    recorder.unsigned_trades += 1
                else:
                    recorder.market(
                        event.ts, FlowType.MB if 2 * event.price > bid + ask else FlowType.MS, event.qty
                    )
        replayer.apply(event)
    return recorder.finish(replayer.report)

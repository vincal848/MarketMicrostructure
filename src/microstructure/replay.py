"""Drive an `OrderBook` from a normalized event stream and audit it.

Replay is the M1 correctness check for the book engine on real data. The
venue reports each execution against a *named* resting order; under
price-time priority that order must be the one at the front of the queue at
its side's best price. `Replayer` applies every event and counts anything
that contradicts that, or that the book cannot explain:

    priority_violations   an 'E' execution of an order that was not first in
                          line at the best price
    unknown_order_refs    a cancel/delete/execute/replace of an order id the
                          book does not hold
    quantity_mismatches   a cancel or execution larger than the order's
                          remaining size
    crossing_adds         a displayed add that would have traded on arrival
                          (the venue would have reported executions instead)

Executions with an explicit price (ITCH 'C', used for crosses) are applied
but not priority-audited, since crosses legitimately ignore queue order.
Hidden trades are counted and leave the displayed book alone.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from microstructure.book import Depth, OrderBook, Side
from microstructure.events import (
    AddOrder,
    CancelOrder,
    DeleteOrder,
    ExecuteOrder,
    ExecuteOrderWithPrice,
    HiddenTrade,
    OrderEvent,
    ReplaceOrder,
    SystemEvent,
    seed_order_id,
)

MARKET_OPEN, MARKET_CLOSE = "Q", "M"


@dataclass
class ReplayReport:
    events: int = 0
    executions_audited: int = 0
    priority_violations: int = 0
    executions_with_price: int = 0
    unknown_order_refs: int = 0
    quantity_mismatches: int = 0
    crossing_adds: int = 0
    hidden_trades: int = 0
    first_issues: list[str] = field(default_factory=list)
    max_issues_kept: int = 20

    def is_clean(self) -> bool:
        problems = self.priority_violations + self.unknown_order_refs
        return problems + self.quantity_mismatches + self.crossing_adds == 0

    def record(self, message: str) -> None:
        if len(self.first_issues) < self.max_issues_kept:
            self.first_issues.append(message)


class Replayer:
    """Applies events one at a time to `book`, accumulating a `ReplayReport`.

    With `audit_continuous_only`, auditing is switched on by the market-open
    system event ('Q') and off by market close ('M'); pre-market and auction
    activity is still applied to the book, just not judged.
    """

    def __init__(self, book: OrderBook | None = None, *, audit_continuous_only: bool = False) -> None:
        self.book = book if book is not None else OrderBook()
        self.report = ReplayReport()
        self._continuous_only = audit_continuous_only
        self._auditing = not audit_continuous_only

    def apply(self, event: OrderEvent) -> None:
        self.report.events += 1
        match event:
            case SystemEvent():
                self._on_system(event)
            case AddOrder():
                self._add(event.ts, event.order_id, event.side, event.price, event.qty)
            case CancelOrder():
                self._cancel(event)
            case DeleteOrder():
                self._delete(event)
            case ReplaceOrder():
                self._replace(event)
            case ExecuteOrder():
                self._execute(event.ts, event.order_id, event.qty, audit=self._auditing)
            case ExecuteOrderWithPrice():
                self.report.executions_with_price += 1
                self._execute(event.ts, event.order_id, event.qty, audit=False)
            case HiddenTrade():
                self.report.hidden_trades += 1

    def _on_system(self, event: SystemEvent) -> None:
        if self._continuous_only and event.code == MARKET_OPEN:
            self._auditing = True
        elif self._continuous_only and event.code == MARKET_CLOSE:
            self._auditing = False

    def _add(self, ts: int, order_id: int, side: Side, price: int, qty: int) -> None:
        if self._auditing and self.book.would_cross(side, price):
            self.report.crossing_adds += 1
            self.report.record(f"ts={ts}: add of order {order_id} at {price} crosses the book")
        self.book.add_limit_order(order_id, side, price, qty)

    def _unknown(self, ts: int, order_id: int, action: str) -> None:
        self.report.unknown_order_refs += 1
        self.report.record(f"ts={ts}: {action} of unknown order {order_id}")

    def _cancel(self, event: CancelOrder) -> None:
        if event.order_id not in self.book:
            self._unknown(event.ts, event.order_id, "cancel")
            return
        resting = self.book.resting_order(event.order_id).qty
        if event.qty > resting:
            self.report.quantity_mismatches += 1
            self.report.record(
                f"ts={event.ts}: cancel of {event.qty} > resting {resting} on {event.order_id}"
            )
        self.book.cancel_order(event.order_id, min(event.qty, resting))

    def _delete(self, event: DeleteOrder) -> None:
        if event.order_id not in self.book:
            self._unknown(event.ts, event.order_id, "delete")
            return
        self.book.cancel_order(event.order_id)

    def _replace(self, event: ReplaceOrder) -> None:
        if event.order_id not in self.book:
            self._unknown(event.ts, event.order_id, "replace")
            return
        side = self.book.resting_order(event.order_id).side
        self.book.cancel_order(event.order_id)
        self._add(event.ts, event.new_order_id, side, event.price, event.qty)

    def _execute(self, ts: int, order_id: int, qty: int, *, audit: bool) -> None:
        if order_id not in self.book:
            self._unknown(ts, order_id, "execution")
            return
        if audit:
            self.report.executions_audited += 1
            if not self.book.is_at_front_of_best(order_id):
                self.report.priority_violations += 1
                ahead = self.book.queue_ahead(order_id)
                self.report.record(
                    f"ts={ts}: executed order {order_id} was not first at the best ({ahead} ahead)"
                )
        resting = self.book.resting_order(order_id).qty
        if qty > resting:
            self.report.quantity_mismatches += 1
            self.report.record(f"ts={ts}: execution of {qty} > resting {resting} on {order_id}")
        self.book.execute_order(order_id, min(qty, resting))


def replay(
    events: Iterable[OrderEvent], *, book: OrderBook | None = None, audit_continuous_only: bool = False
) -> Replayer:
    """Apply every event and return the replayer (its `book` and `report`)."""
    replayer = Replayer(book, audit_continuous_only=audit_continuous_only)
    for event in events:
        replayer.apply(event)
    return replayer


def seed_book(depth: Depth) -> OrderBook:
    """A book holding one synthetic order (see `events.seed_order_id`) per
    level of `depth`, for replays that start from a snapshot."""
    book = OrderBook()
    bids, asks = depth
    for side, levels in ((Side.BID, bids), (Side.ASK, asks)):
        for price, qty in levels:
            book.add_limit_order(seed_order_id(side, price), side, price, qty)
    return book


def verify_snapshots(
    events: Sequence[OrderEvent], snapshots: Sequence[Depth], book: OrderBook, n_levels: int
) -> list[int]:
    """Replay `events` into `book`, comparing the top `n_levels` against
    `snapshots[i]` after `events[i]`. Returns the indices that differ."""
    if len(events) != len(snapshots):
        raise ValueError(f"{len(events)} events but {len(snapshots)} snapshots")
    replayer = Replayer(book)
    mismatched: list[int] = []
    for i, (event, expected) in enumerate(zip(events, snapshots, strict=True)):
        replayer.apply(event)
        if replayer.book.depth_snapshot(n_levels) != expected:
            mismatched.append(i)
    return mismatched


def depth_at(events: Iterable[OrderEvent], ts: int, n_levels: int) -> Depth:
    """The top `n_levels` after replaying every event strictly before `ts`
    (e.g. the opening book of a simulation window)."""
    replayer = Replayer()
    for event in events:
        if event.ts >= ts:
            break
        replayer.apply(event)
    return replayer.book.depth_snapshot(n_levels)

"""Price-time priority limit order book with integer tick prices.

Orders rest in FIFO queues at each price level. An incoming order executes
against the opposite side, best price first and oldest order first within a
price, before any limit remainder joins the book. So `best_bid() < best_ask()`
holds after every operation, not just after non-marketable adds.

Prices are integer ticks. LOBSTER prices in units of 1/10000 of a dollar (see
`lobster.py`); converting to a tick index is the caller's job, so this module
stays agnostic to the instrument's tick size.

Every public method validates its inputs before touching any state: a rejected
call leaves the book exactly as it was.
"""

from __future__ import annotations

import operator
from bisect import insort
from collections import deque
from dataclasses import dataclass, replace
from enum import StrEnum


class Side(StrEnum):
    """Side of the order being submitted: BID is a buy, ASK is a sell.

    A market order on BID buys, consuming resting asks; LOBSTER's own
    `direction` column uses the same convention for the resting order.
    """

    BID = "bid"
    ASK = "ask"

    @property
    def opposite(self) -> Side:
        return Side.ASK if self is Side.BID else Side.BID


@dataclass(slots=True)
class Order:
    """A resting order. `qty` is reduced in place by partial fills/cancels."""

    order_id: int
    side: Side
    price: int
    qty: int


@dataclass(frozen=True, slots=True)
class Fill:
    """One match between an incoming order and a resting one."""

    price: int
    qty: int
    resting_order_id: int


Level = tuple[int, int]
"""(price, total resting quantity) at one price level."""

Depth = tuple[list[Level], list[Level]]
"""(bid levels, ask levels), each best first."""


class _BookSide:
    """All resting orders on one side, with price levels kept sorted.

    `_prices` is ordered worst to best, so the best price is always
    `_prices[-1]` and consuming the top level is an O(1) `pop()`. "Better"
    means higher for bids and lower for asks; `_rank` folds that into one
    number where larger is always better, which lets every comparison below
    be written once instead of once per side.
    """

    def __init__(self, side: Side) -> None:
        self.side = side
        self._levels: dict[int, deque[Order]] = {}
        self._prices: list[int] = []

    def _rank(self, price: int) -> int:
        return price if self.side is Side.BID else -price

    def __bool__(self) -> bool:
        return bool(self._prices)

    def best(self) -> int | None:
        return self._prices[-1] if self._prices else None

    def is_marketable_against(self, limit_price: int | None) -> bool:
        """Whether an incoming opposite-side order at `limit_price` (None for
        a market order) would trade against this side's best level."""
        if not self._prices:
            return False
        if limit_price is None:
            return True
        return self._rank(self._prices[-1]) >= self._rank(limit_price)

    def append(self, order: Order) -> None:
        queue = self._levels.get(order.price)
        if queue is None:
            queue = self._levels[order.price] = deque()
            insort(self._prices, order.price, key=self._rank)
        queue.append(order)

    def remove(self, order: Order) -> None:
        queue = self._levels[order.price]
        queue.remove(order)
        if not queue:
            self._drop_level(order.price)

    def consume_best(self, qty: int, fills: list[Fill]) -> tuple[int, list[Order]]:
        """Fill up to `qty` FIFO from the best level, appending to `fills`.

        Returns the quantity left unfilled and the orders fully consumed (so
        the caller can forget their ids).
        """
        price = self._prices[-1]
        queue = self._levels[price]
        finished: list[Order] = []
        while qty > 0 and queue:
            resting = queue[0]
            traded = min(qty, resting.qty)
            fills.append(Fill(price, traded, resting.order_id))
            resting.qty -= traded
            qty -= traded
            if resting.qty == 0:
                finished.append(queue.popleft())
        if not queue:
            self._drop_level(price)
        return qty, finished

    def qty_ahead_of(self, order: Order) -> int:
        ahead = 0
        for queued in self._levels[order.price]:
            if queued is order:
                return ahead
            ahead += queued.qty
        raise AssertionError(f"order {order.order_id} missing from its level")

    def is_at_front_of_best(self, order: Order) -> bool:
        return order.price == self._prices[-1] and self._levels[order.price][0] is order

    def top_levels(self, n_levels: int) -> list[Level]:
        best_first = reversed(self._prices[-n_levels:]) if n_levels > 0 else []
        return [(p, sum(o.qty for o in self._levels[p])) for p in best_first]

    def _drop_level(self, price: int) -> None:
        del self._levels[price]
        if self._prices[-1] == price:
            self._prices.pop()
        else:
            self._prices.remove(price)


def _positive_qty(qty: int) -> int:
    qty = operator.index(qty)
    if qty <= 0:
        raise ValueError(f"qty must be positive, got {qty!r}")
    return qty


class OrderBook:
    """Price-time priority book for a single instrument."""

    def __init__(self) -> None:
        self._sides = {Side.BID: _BookSide(Side.BID), Side.ASK: _BookSide(Side.ASK)}
        self._orders: dict[int, Order] = {}

    def __contains__(self, order_id: object) -> bool:
        return order_id in self._orders

    def resting_order(self, order_id: int) -> Order:
        """A copy of the resting order; mutating it does not touch the book."""
        return replace(self._orders[order_id])

    def queue_ahead(self, order_id: int) -> int:
        """Shares resting ahead of `order_id` at its own price."""
        order = self._orders[order_id]
        return self._sides[order.side].qty_ahead_of(order)

    def is_at_front_of_best(self, order_id: int) -> bool:
        """Whether `order_id` is first in the queue at its side's best price,
        i.e. the next order an incoming market order would hit."""
        order = self._orders[order_id]
        return self._sides[order.side].is_at_front_of_best(order)

    def would_cross(self, side: Side, price: int) -> bool:
        """Whether a limit order on `side` at `price` would trade on arrival."""
        return self._sides[Side(side).opposite].is_marketable_against(price)

    def best_bid(self) -> int | None:
        return self._sides[Side.BID].best()

    def best_ask(self) -> int | None:
        return self._sides[Side.ASK].best()

    def add_limit_order(self, order_id: int, side: Side, price: int, qty: int) -> list[Fill]:
        """Submit a limit order: trade whatever crosses, rest the remainder.

        Returns the fills from the crossing part in execution order; an
        unfilled remainder rests at `price`.
        """
        side = Side(side)
        price = operator.index(price)
        qty = _positive_qty(qty)
        if order_id in self._orders:
            raise ValueError(f"duplicate order_id {order_id!r}")

        fills, remaining = self._match(side, qty, limit_price=price)
        if remaining > 0:
            order = Order(order_id, side, price, remaining)
            self._sides[side].append(order)
            self._orders[order_id] = order
        return fills

    def market_order(self, side: Side, qty: int) -> tuple[list[Fill], int]:
        """Submit a market order, walking the opposite side until filled.

        Returns (fills, leftover); leftover > 0 means the book ran dry first.
        """
        return self._match(Side(side), _positive_qty(qty), limit_price=None)

    def cancel_order(self, order_id: int, qty: int | None = None) -> None:
        """Cancel a resting order in full (`qty=None`) or by `qty` shares.

        A `qty` at or above the resting quantity cancels the whole order.
        Raises KeyError if `order_id` is not resting on the book.
        """
        order = self._orders[order_id]
        if qty is not None:
            qty = _positive_qty(qty)
        if qty is None or qty >= order.qty:
            self._sides[order.side].remove(order)
            del self._orders[order_id]
        else:
            order.qty -= qty

    def execute_order(self, order_id: int, qty: int) -> Fill:
        """Trade `qty` shares of a specific resting order at its price.

        This is the exchange's view of a match: the venue names the resting
        order that traded, which need not be the one `market_order` would
        pick (replay audits that separately). Partial executions keep the
        order's queue position.
        """
        order = self._orders[order_id]
        qty = _positive_qty(qty)
        if qty > order.qty:
            raise ValueError(f"execution of {qty} exceeds order {order_id}'s resting {order.qty}")
        if qty == order.qty:
            self._sides[order.side].remove(order)
            del self._orders[order_id]
        else:
            order.qty -= qty
        return Fill(order.price, qty, order_id)

    def depth_snapshot(self, n_levels: int) -> Depth:
        """Top `n_levels` per side as (price, total_qty), best first."""
        return (
            self._sides[Side.BID].top_levels(n_levels),
            self._sides[Side.ASK].top_levels(n_levels),
        )

    def _match(self, side: Side, qty: int, limit_price: int | None) -> tuple[list[Fill], int]:
        resting = self._sides[side.opposite]
        fills: list[Fill] = []
        while qty > 0 and resting.is_marketable_against(limit_price):
            qty, finished = resting.consume_best(qty, fills)
            for order in finished:
                del self._orders[order.order_id]
        return fills, qty

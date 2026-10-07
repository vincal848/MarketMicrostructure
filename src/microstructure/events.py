"""Normalized order events: the one vocabulary every data source maps onto.

Adapters (`itch`, `lobster`) convert their native messages into these
types at the boundary, so replay, classification and calibration never see
source-specific formats. All timestamps are integer nanoseconds since
midnight; all prices are integers in the source's native unit (1/10000 of
a dollar for both ITCH and LOBSTER).

`seed_order_id` names the synthetic order that stands in for all volume
resting at a price level when replay starts from a snapshot rather than an
empty book (see `replay.seed_book`).
"""

from __future__ import annotations

from dataclasses import dataclass

from microstructure.book import Side


def _check_ts(ts: int) -> None:
    if ts < 0:
        raise ValueError(f"ts must be non-negative nanoseconds, got {ts!r}")


def _check_qty(qty: int) -> None:
    if qty <= 0:
        raise ValueError(f"qty must be positive, got {qty!r}")


@dataclass(frozen=True, slots=True)
class AddOrder:
    """A new displayed limit order rests on the book."""

    ts: int
    order_id: int
    side: Side
    price: int
    qty: int

    def __post_init__(self) -> None:
        _check_ts(self.ts)
        _check_qty(self.qty)


@dataclass(frozen=True, slots=True)
class CancelOrder:
    """`qty` shares of a resting order are cancelled; the rest keeps priority."""

    ts: int
    order_id: int
    qty: int

    def __post_init__(self) -> None:
        _check_ts(self.ts)
        _check_qty(self.qty)


@dataclass(frozen=True, slots=True)
class DeleteOrder:
    """A resting order is removed entirely."""

    ts: int
    order_id: int

    def __post_init__(self) -> None:
        _check_ts(self.ts)


@dataclass(frozen=True, slots=True)
class ReplaceOrder:
    """Cancel `order_id` and add `new_order_id` on the same side at the back
    of the queue (an exchange replace never keeps time priority)."""

    ts: int
    order_id: int
    new_order_id: int
    price: int
    qty: int

    def __post_init__(self) -> None:
        _check_ts(self.ts)
        _check_qty(self.qty)


@dataclass(frozen=True, slots=True)
class ExecuteOrder:
    """`qty` shares of a resting order trade at its own limit price."""

    ts: int
    order_id: int
    qty: int

    def __post_init__(self) -> None:
        _check_ts(self.ts)
        _check_qty(self.qty)


@dataclass(frozen=True, slots=True)
class ExecuteOrderWithPrice:
    """`qty` shares of a resting order trade at `price`, which may differ
    from its limit (ITCH 'C': crosses and other non-FIFO executions).
    `printable` is False when the volume is printed elsewhere, e.g. by the
    cross's own trade message."""

    ts: int
    order_id: int
    qty: int
    price: int
    printable: bool

    def __post_init__(self) -> None:
        _check_ts(self.ts)
        _check_qty(self.qty)


@dataclass(frozen=True, slots=True)
class HiddenTrade:
    """An execution against non-displayed liquidity: a trade with no effect
    on the visible book. `resting_side` is as reported by the source; ITCH
    publishes it unreliably, so classification infers the aggressor from the
    trade price instead."""

    ts: int
    resting_side: Side
    price: int
    qty: int

    def __post_init__(self) -> None:
        _check_ts(self.ts)
        _check_qty(self.qty)


@dataclass(frozen=True, slots=True)
class SystemEvent:
    """A session boundary. ITCH codes: 'O' start of messages, 'S' start of
    system hours, 'Q' start of market hours, 'M' end of market hours,
    'E' end of system hours, 'C' end of messages."""

    ts: int
    code: str

    def __post_init__(self) -> None:
        _check_ts(self.ts)


OrderEvent = (
    AddOrder
    | CancelOrder
    | DeleteOrder
    | ReplaceOrder
    | ExecuteOrder
    | ExecuteOrderWithPrice
    | HiddenTrade
    | SystemEvent
)


def seed_order_id(side: Side, price: int) -> int:
    """Id of the synthetic order holding a seeded level's volume.

    Negative, so it can never collide with an exchange-assigned id, and
    unique per (side, price).
    """
    return -(2 * price + (0 if side is Side.BID else 1)) - 1

"""Parser for LOBSTER message and orderbook CSV files.

LOBSTER (https://lobsterdata.com) reconstructs these from raw NASDAQ ITCH.
Both files have no header row and are aligned row for row: message row k is
the event that produced orderbook row k. Real sample files live under data/
(gitignored); the tests use a small hand-made fixture instead.

Message file columns (6), in order:
    time        seconds after midnight (decimal, nanosecond precision)
    type        see `EventType`
    order_id    unique while the order is live; LOBSTER may reuse an id
                once it is fully gone
    size        shares
    price       dollars x 10000, as an integer
    direction   see `Direction` -- the *resting* limit order's side, so an
                aggressive market buy appears as an execution (type 4 or 5)
                of a sell limit order (direction -1)

Orderbook file columns (4 x n_levels), for level i = 1 .. n_levels:
    ask_price_i, ask_size_i, bid_price_i, bid_size_i
Level 1 is the best bid/ask. An empty level is coded as price +-9999999999
with size 0; rows like that are passed through unchanged.

`to_events` converts messages to the normalized event model (`events.py`).
A LOBSTER window opens with orders already resting, so replay starts from
a snapshot (`replay.seed_book`) and messages about orders the window never
added are redirected to that level's synthetic seed order.

Column counts are checked against the raw file *before* names are attached.
pandas, given fewer names than columns, silently moves the surplus leading
columns into the index -- so reading a 2-level file as 1 level would return
level 2 labelled as level 1 with no error.
"""

from __future__ import annotations

import os
from enum import IntEnum

import numpy as np
import pandas as pd

from microstructure.book import Depth, Level, Side
from microstructure.events import (
    AddOrder,
    CancelOrder,
    DeleteOrder,
    ExecuteOrder,
    HiddenTrade,
    OrderEvent,
    SystemEvent,
    seed_order_id,
)

FilePath = str | os.PathLike[str]

MESSAGE_COLUMNS = ["time", "type", "order_id", "size", "price", "direction"]

EMPTY_LEVEL_PRICE = 9_999_999_999
"""Absolute price LOBSTER writes for a level that does not exist."""


class EventType(IntEnum):
    NEW_LIMIT_ORDER = 1
    PARTIAL_CANCELLATION = 2
    FULL_DELETION = 3
    VISIBLE_EXECUTION = 4
    HIDDEN_EXECUTION = 5
    CROSS_TRADE = 6
    TRADING_HALT = 7


class Direction(IntEnum):
    BUY = 1
    SELL = -1


def _read_headerless(path: FilePath, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(columns):
        raise ValueError(f"{os.fspath(path)}: expected {len(columns)} columns, file has {df.shape[1]}")
    df.columns = pd.Index(columns)
    return df


def _reject_unknown(df: pd.DataFrame, column: str, allowed: type[IntEnum]) -> None:
    unknown = set(df[column]) - {int(member) for member in allowed}
    if unknown:
        raise ValueError(f"unknown LOBSTER {column} value(s): {sorted(unknown)}")


def read_messages(path: FilePath) -> pd.DataFrame:
    """Read a message file into a DataFrame with `MESSAGE_COLUMNS`."""
    df = _read_headerless(path, MESSAGE_COLUMNS)
    _reject_unknown(df, "type", EventType)
    _reject_unknown(df, "direction", Direction)
    return df


def orderbook_columns(n_levels: int) -> list[str]:
    """Column names for an n_levels orderbook file, best level first."""
    if n_levels < 1:
        raise ValueError(f"n_levels must be at least 1, got {n_levels!r}")
    return [
        f"{field}_{level}"
        for level in range(1, n_levels + 1)
        for field in ("ask_price", "ask_size", "bid_price", "bid_size")
    ]


def read_orderbook(path: FilePath, n_levels: int) -> pd.DataFrame:
    """Read an orderbook file into a DataFrame named by `orderbook_columns`."""
    return _read_headerless(path, orderbook_columns(n_levels))


def read_paired(
    message_path: FilePath, orderbook_path: FilePath, n_levels: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both files and check they are aligned row for row."""
    messages = read_messages(message_path)
    book = read_orderbook(orderbook_path, n_levels)
    if len(messages) != len(book):
        raise ValueError(
            f"message file has {len(messages)} rows, orderbook file has {len(book)}; "
            "they must be aligned row for row"
        )
    return messages, book


def to_events(messages: pd.DataFrame, first_row: int = 0) -> list[OrderEvent]:
    """Convert message rows `first_row` onward into normalized events, one
    event per row so they stay aligned with the orderbook file.

    Orders not added within those rows (they were resting when the window,
    or the replay, began) are redirected to the seed order for their side
    and price. A full deletion of such an order becomes a partial cancel of
    the seed by the order's own size. Cross trades (type 6) and halts
    (type 7) leave the book alone and become `SystemEvent`s coded "cross"
    and "halt".
    """
    known: set[int] = set()
    events: list[OrderEvent] = []
    rows = messages.iloc[first_row:]
    for time, kind, order_id, size, price, direction in zip(
        rows["time"],
        rows["type"],
        rows["order_id"],
        rows["size"],
        rows["price"],
        rows["direction"],
        strict=True,
    ):
        ts = round(float(time) * 1e9)
        side = Side.BID if direction == Direction.BUY else Side.ASK
        oid, qty, px = int(order_id), int(size), int(price)
        target = oid if oid in known else seed_order_id(side, px)
        match EventType(kind):
            case EventType.NEW_LIMIT_ORDER:
                known.add(oid)
                events.append(AddOrder(ts=ts, order_id=oid, side=side, price=px, qty=qty))
            case EventType.PARTIAL_CANCELLATION:
                events.append(CancelOrder(ts=ts, order_id=target, qty=qty))
            case EventType.FULL_DELETION if oid in known:
                known.discard(oid)
                events.append(DeleteOrder(ts=ts, order_id=oid))
            case EventType.FULL_DELETION:
                events.append(CancelOrder(ts=ts, order_id=target, qty=qty))
            case EventType.VISIBLE_EXECUTION:
                events.append(ExecuteOrder(ts=ts, order_id=target, qty=qty))
            case EventType.HIDDEN_EXECUTION:
                events.append(HiddenTrade(ts=ts, resting_side=side, price=px, qty=qty))
            case EventType.CROSS_TRADE:
                events.append(SystemEvent(ts=ts, code="cross"))
            case EventType.TRADING_HALT:
                events.append(SystemEvent(ts=ts, code="halt"))
    return events


def depths(orderbook: pd.DataFrame, n_levels: int) -> list[Depth]:
    """Each orderbook row as (bid levels, ask levels), best first, with
    LOBSTER's empty-level placeholders dropped."""
    values = orderbook[orderbook_columns(n_levels)].to_numpy(dtype=np.int64)
    ask_px, ask_sz = values[:, 0::4], values[:, 1::4]
    bid_px, bid_sz = values[:, 2::4], values[:, 3::4]

    def side_levels(prices: np.ndarray, sizes: np.ndarray) -> list[Level]:
        return [
            (int(p), int(q))
            for p, q in zip(prices, sizes, strict=True)
            if q > 0 and abs(int(p)) != EMPTY_LEVEL_PRICE
        ]

    return [
        (side_levels(bid_px[row], bid_sz[row]), side_levels(ask_px[row], ask_sz[row]))
        for row in range(values.shape[0])
    ]

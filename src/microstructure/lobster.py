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

Column counts are checked against the raw file *before* names are attached.
pandas, given fewer names than columns, silently moves the surplus leading
columns into the index -- so reading a 2-level file as 1 level would return
level 2 labelled as level 1 with no error.
"""

from __future__ import annotations

import os
from enum import IntEnum

import pandas as pd

FilePath = str | os.PathLike[str]

MESSAGE_COLUMNS = ["time", "type", "order_id", "size", "price", "direction"]


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

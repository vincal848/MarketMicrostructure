"""Parser for LOBSTER's message and orderbook CSV files.

LOBSTER (https://lobsterdata.com) reconstructs these from raw NASDAQ ITCH
data. Both files have no header row and are aligned row for row: message
file row k is the event that produced orderbook file row k's snapshot. See
README > Data for where the actual sample files are expected to live
(data/, gitignored) -- the tests here run against a small hand-made fixture
instead.

Message file columns (6), in order:
    time        seconds after midnight (decimal, nanosecond precision)
    type        1 new limit order, 2 partial cancellation, 3 full deletion,
                4 visible execution, 5 hidden execution, 6 cross trade,
                7 trading halt
    order_id    unique while the order is live; LOBSTER may reuse an id
                once it is fully gone
    size        shares
    price       dollars x 10000, as an integer
    direction   1 buy, -1 sell -- the resting limit order's side. An
                aggressive market buy therefore shows up as an execution
                (type 4 or 5) of a sell limit order (direction -1).

Orderbook file columns (4 x n_levels), in order, for level i = 1 .. n_levels:
    ask_price_i, ask_size_i, bid_price_i, bid_size_i
Level 1 is the best bid/ask. LOBSTER codes an empty level as price
+-9999999999 with size 0; this parser passes such rows through unchanged,
since nothing here needs to special-case them yet.
"""

import pandas as pd

MESSAGE_COLUMNS = ["time", "type", "order_id", "size", "price", "direction"]

EVENT_TYPES = {
    1: "new_limit_order",
    2: "partial_cancellation",
    3: "full_deletion",
    4: "visible_execution",
    5: "hidden_execution",
    6: "cross_trade",
    7: "trading_halt",
}


def read_messages(path):
    """Read a LOBSTER message file into a DataFrame with MESSAGE_COLUMNS."""
    df = pd.read_csv(path, header=None, names=MESSAGE_COLUMNS)
    unknown = set(df["type"]) - set(EVENT_TYPES)
    if unknown:
        raise ValueError("unknown LOBSTER event type(s): %s" % sorted(unknown))
    return df


def orderbook_columns(n_levels):
    """Column names for an n_levels orderbook file, best level first."""
    columns = []
    for level in range(1, n_levels + 1):
        columns += [
            "ask_price_%d" % level, "ask_size_%d" % level,
            "bid_price_%d" % level, "bid_size_%d" % level,
        ]
    return columns


def read_orderbook(path, n_levels):
    """Read a LOBSTER orderbook file into a DataFrame with 4 * n_levels
    columns, named as in `orderbook_columns`.
    """
    columns = orderbook_columns(n_levels)
    df = pd.read_csv(path, header=None, names=columns)
    if len(df.columns) != len(columns):
        raise ValueError(
            "expected %d columns for %d levels, file has %d" % (
                len(columns), n_levels, len(df.columns)))
    return df


def read_paired(message_path, orderbook_path, n_levels):
    """Read both files and check they are aligned row for row.

    Returns (messages, orderbook) DataFrames.
    """
    messages = read_messages(message_path)
    book = read_orderbook(orderbook_path, n_levels)
    if len(messages) != len(book):
        raise ValueError(
            "message file has %d rows, orderbook file has %d -- they "
            "should be aligned row for row" % (len(messages), len(book)))
    return messages, book

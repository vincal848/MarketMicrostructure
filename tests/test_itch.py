"""Nasdaq TotalView-ITCH 5.0 decoding into the normalized event model."""

import gzip
import io
from pathlib import Path

import itch_writer as w
import pytest

from microstructure.book import Side
from microstructure.events import (
    AddOrder,
    CancelOrder,
    DeleteOrder,
    ExecuteOrder,
    ExecuteOrderWithPrice,
    HiddenTrade,
    ReplaceOrder,
    SystemEvent,
)
from microstructure.itch import read_itch

SPY, AAPL = 7, 9


def _day() -> bytes:
    return w.frame(
        w.system_event(1, "O"),
        w.stock_directory(AAPL, "AAPL"),
        w.stock_directory(SPY, "SPY"),
        w.system_event(2, "Q"),
        w.add_order(SPY, 10, ref=1, side="B", shares=100, symbol="SPY", price=2_650_000),
        w.add_order(AAPL, 11, ref=2, side="S", shares=50, symbol="AAPL", price=1_500_000),
        w.add_order_mpid(SPY, 12, ref=3, side="S", shares=200, symbol="SPY", price=2_650_100),
        w.stock_trading_action(SPY, 13, "SPY"),
        w.order_executed(SPY, 14, ref=3, shares=50),
        w.order_executed_with_price(SPY, 15, ref=3, shares=10, price=2_650_050, printable=True),
        w.order_cancel(SPY, 16, ref=1, shares=40),
        w.order_replace(SPY, 17, ref=1, new_ref=4, shares=60, price=2_649_900),
        w.trade(SPY, 18, side="B", shares=25, symbol="SPY", price=2_650_050),
        w.order_delete(SPY, 19, ref=4),
        w.order_delete(AAPL, 20, ref=2),
        w.system_event(21, "M"),
    )


EXPECTED_SPY = [
    SystemEvent(ts=1, code="O"),
    SystemEvent(ts=2, code="Q"),
    AddOrder(ts=10, order_id=1, side=Side.BID, price=2_650_000, qty=100),
    AddOrder(ts=12, order_id=3, side=Side.ASK, price=2_650_100, qty=200),
    ExecuteOrder(ts=14, order_id=3, qty=50),
    ExecuteOrderWithPrice(ts=15, order_id=3, qty=10, price=2_650_050, printable=True),
    CancelOrder(ts=16, order_id=1, qty=40),
    ReplaceOrder(ts=17, order_id=1, new_order_id=4, price=2_649_900, qty=60),
    HiddenTrade(ts=18, resting_side=Side.BID, price=2_650_050, qty=25),
    DeleteOrder(ts=19, order_id=4),
    SystemEvent(ts=21, code="M"),
]


def test_decodes_every_supported_message_for_one_symbol() -> None:
    assert list(read_itch(io.BytesIO(_day()), "SPY")) == EXPECTED_SPY


def test_symbol_filter_isolates_the_other_stock() -> None:
    aapl_orders = [e for e in read_itch(io.BytesIO(_day()), "AAPL") if not isinstance(e, SystemEvent)]
    assert aapl_orders == [
        AddOrder(ts=11, order_id=2, side=Side.ASK, price=1_500_000, qty=50),
        DeleteOrder(ts=20, order_id=2),
    ]


def test_reads_gzipped_files_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "day.NASDAQ_ITCH50.gz"
    path.write_bytes(gzip.compress(_day()))
    assert list(read_itch(path, "SPY")) == EXPECTED_SPY


def test_reads_uncompressed_files_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "day.NASDAQ_ITCH50"
    path.write_bytes(_day())
    assert list(read_itch(path, "SPY")) == EXPECTED_SPY


def test_large_timestamps_use_all_six_bytes() -> None:
    four_pm = 16 * 3600 * 10**9
    stream = w.frame(w.stock_directory(SPY, "SPY"), w.order_delete(SPY, four_pm, ref=1))
    assert list(read_itch(io.BytesIO(stream), "SPY")) == [DeleteOrder(ts=four_pm, order_id=1)]


def test_unknown_symbol_is_an_error_not_an_empty_stream() -> None:
    with pytest.raises(ValueError, match="MSFT"):
        list(read_itch(io.BytesIO(_day()), "MSFT"))


def test_truncated_stream_is_an_error() -> None:
    with pytest.raises(ValueError, match="truncated"):
        list(read_itch(io.BytesIO(_day()[:-3]), "SPY"))


def test_chunk_boundaries_do_not_split_messages() -> None:
    # A tiny read size forces every message to straddle a chunk boundary.
    assert list(read_itch(io.BytesIO(_day()), "SPY", chunk_size=7)) == EXPECTED_SPY

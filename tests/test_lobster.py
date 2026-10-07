"""LOBSTER message/orderbook CSV parsing against a tiny hand-made fixture."""

from pathlib import Path

import pytest

from microstructure.lobster import (
    MESSAGE_COLUMNS,
    EventType,
    depths,
    read_messages,
    read_orderbook,
    read_paired,
    to_events,
)

FIXTURES = Path(__file__).parent / "fixtures"
MESSAGES = FIXTURES / "tiny_message.csv"
ORDERBOOK = FIXTURES / "tiny_orderbook.csv"


def test_read_messages_parses_columns_and_event_types() -> None:
    df = read_messages(MESSAGES)
    assert list(df.columns) == MESSAGE_COLUMNS
    assert len(df) == 5
    assert df.loc[0, "type"] == EventType.NEW_LIMIT_ORDER
    assert df.loc[0, "price"] == 585000
    assert df.loc[3, "type"] == EventType.VISIBLE_EXECUTION  # order 2 executed


def test_read_orderbook_parses_n_levels() -> None:
    df = read_orderbook(ORDERBOOK, n_levels=1)
    assert list(df.columns) == ["ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"]
    assert len(df) == 5
    assert df.loc[1, "ask_price_1"] == 585100
    assert df.loc[1, "bid_size_1"] == 100


def test_read_orderbook_rejects_a_level_count_that_does_not_match_the_file(tmp_path: Path) -> None:
    # Regression: pandas used to move the surplus columns into the index,
    # returning level 2 labelled as level 1 without any error.
    two_levels = tmp_path / "two_level_orderbook.csv"
    two_levels.write_text("585100,50,585000,100,585200,10,584900,20\n")
    with pytest.raises(ValueError, match="expected 4 columns, file has 8"):
        read_orderbook(two_levels, n_levels=1)


def test_read_paired_checks_row_alignment(tmp_path: Path) -> None:
    messages, book = read_paired(MESSAGES, ORDERBOOK, n_levels=1)
    assert len(messages) == len(book) == 5

    short_book = tmp_path / "short_orderbook.csv"
    short_book.write_text("9999999999,0,585000,100\n")
    with pytest.raises(ValueError, match="aligned"):
        read_paired(MESSAGES, short_book, n_levels=1)


def test_read_messages_rejects_an_unknown_event_type(tmp_path: Path) -> None:
    bad = tmp_path / "bad_message.csv"
    bad.write_text("34200.1,99,1,100,585000,1\n")
    with pytest.raises(ValueError, match="type"):
        read_messages(bad)


def test_read_messages_rejects_an_unknown_direction(tmp_path: Path) -> None:
    bad = tmp_path / "bad_direction.csv"
    bad.write_text("34200.1,1,1,100,585000,0\n")
    with pytest.raises(ValueError, match="direction"):
        read_messages(bad)


# --- Phase 1: LOBSTER messages -> normalized events --------------------------


def test_to_events_maps_each_message_type() -> None:
    from microstructure.book import Side
    from microstructure.events import AddOrder, DeleteOrder, ExecuteOrder

    events = to_events(read_messages(MESSAGES))
    assert events[0] == AddOrder(ts=34_200_100_000_000, order_id=1, side=Side.BID, price=585000, qty=100)
    assert events[3] == ExecuteOrder(ts=34_200_400_000_000, order_id=2, qty=50)
    assert events[4] == DeleteOrder(ts=34_200_500_000_000, order_id=3)


def test_to_events_maps_pre_window_orders_onto_seed_orders() -> None:
    from microstructure.book import Side
    from microstructure.events import CancelOrder, ExecuteOrder, seed_order_id

    events = to_events(read_messages(FIXTURES / "seeded_message.csv"), first_row=1)
    # Row 1 executes order 900, never added inside the window.
    assert events[0] == ExecuteOrder(ts=34_200_200_000_000, order_id=seed_order_id(Side.ASK, 585100), qty=60)
    # A full deletion of an unseen order removes only its own shares from the seed.
    assert events[1] == CancelOrder(ts=34_200_300_000_000, order_id=seed_order_id(Side.BID, 584900), qty=100)
    # Order 10 was added in row 0, i.e. before the replay starts, so it is part of the seed too.
    assert events[2] == CancelOrder(ts=34_200_400_000_000, order_id=seed_order_id(Side.BID, 585000), qty=10)


def test_depths_drop_empty_levels() -> None:
    snapshots = depths(read_orderbook(ORDERBOOK, n_levels=1), n_levels=1)
    assert snapshots[0] == ([(585000, 100)], [])
    assert snapshots[1] == ([(585000, 100)], [(585100, 50)])

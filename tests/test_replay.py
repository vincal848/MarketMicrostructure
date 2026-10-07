"""Replaying order events through the book, with a price-time priority audit."""

from pathlib import Path

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
    SystemEvent,
)
from microstructure.lobster import depths, read_paired, to_events
from microstructure.replay import replay, seed_book, verify_snapshots

FIXTURES = Path(__file__).parent / "fixtures"


def _two_sided() -> list[OrderEvent]:
    return [
        AddOrder(ts=1, order_id=1, side=Side.BID, price=100, qty=10),
        AddOrder(ts=2, order_id=2, side=Side.BID, price=100, qty=10),
        AddOrder(ts=3, order_id=3, side=Side.BID, price=99, qty=10),
        AddOrder(ts=4, order_id=4, side=Side.ASK, price=101, qty=10),
    ]


def test_replay_builds_the_expected_book() -> None:
    events = [*_two_sided(), CancelOrder(ts=5, order_id=3, qty=4), DeleteOrder(ts=6, order_id=4)]
    result = replay(events)
    assert result.book.depth_snapshot(5) == ([(100, 20), (99, 6)], [])
    assert result.report.events == 6
    assert result.report.is_clean()


def test_execution_of_the_queue_head_at_the_best_price_is_clean() -> None:
    result = replay([*_two_sided(), ExecuteOrder(ts=5, order_id=1, qty=10)])
    assert result.report.executions_audited == 1
    assert result.report.priority_violations == 0


def test_execution_behind_the_queue_head_is_a_priority_violation() -> None:
    result = replay([*_two_sided(), ExecuteOrder(ts=5, order_id=2, qty=5)])
    assert result.report.priority_violations == 1
    assert "order 2" in result.report.first_issues[0]


def test_execution_away_from_the_best_price_is_a_priority_violation() -> None:
    result = replay([*_two_sided(), ExecuteOrder(ts=5, order_id=3, qty=5)])
    assert result.report.priority_violations == 1


def test_unknown_order_reference_is_counted_not_raised() -> None:
    result = replay([*_two_sided(), DeleteOrder(ts=5, order_id=42), ExecuteOrder(ts=6, order_id=43, qty=1)])
    assert result.report.unknown_order_refs == 2
    assert not result.report.is_clean()


def test_replace_keeps_side_and_loses_priority() -> None:
    events = [*_two_sided(), ReplaceOrder(ts=5, order_id=1, new_order_id=9, price=100, qty=10)]
    result = replay(events)
    assert result.book.resting_order(9).side is Side.BID
    assert result.book.queue_ahead(9) == 10  # now behind order 2
    assert 1 not in result.book


def test_crossing_add_is_counted() -> None:
    result = replay([*_two_sided(), AddOrder(ts=5, order_id=8, side=Side.BID, price=101, qty=3)])
    assert result.report.crossing_adds == 1


def test_hidden_trades_do_not_touch_the_book() -> None:
    result = replay([*_two_sided(), HiddenTrade(ts=5, resting_side=Side.BID, price=100, qty=50)])
    assert result.report.hidden_trades == 1
    assert result.book.depth_snapshot(1) == ([(100, 20)], [(101, 10)])


def test_executions_with_price_are_applied_but_not_priority_audited() -> None:
    # Opening/closing-cross executions arrive as ITCH 'C' messages and are
    # legitimately out of FIFO order.
    events = [*_two_sided(), ExecuteOrderWithPrice(ts=5, order_id=3, qty=10, price=100, printable=False)]
    result = replay(events)
    assert result.report.priority_violations == 0
    assert result.report.executions_with_price == 1
    assert 3 not in result.book


def test_audit_can_be_restricted_to_continuous_trading() -> None:
    pre_open = ExecuteOrder(ts=5, order_id=2, qty=1)  # out of priority, but before 'Q'
    events: list[OrderEvent] = [*_two_sided(), pre_open, SystemEvent(ts=6, code="Q")]
    events += [ExecuteOrder(ts=7, order_id=2, qty=1), SystemEvent(ts=8, code="M")]
    events += [ExecuteOrder(ts=9, order_id=2, qty=1)]
    result = replay(events, audit_continuous_only=True)
    assert result.report.executions_audited == 1
    assert result.report.priority_violations == 1


def test_crossed_book_inside_the_session_is_counted() -> None:
    result = replay([*_two_sided(), AddOrder(ts=5, order_id=8, side=Side.ASK, price=99, qty=50)])
    assert result.report.crossing_adds == 1


def test_seed_book_rebuilds_depth() -> None:
    depth = ([(100, 30), (99, 5)], [(101, 7)])
    assert seed_book(depth).depth_snapshot(5) == depth


def test_lobster_fixture_replays_to_every_snapshot_exactly() -> None:
    messages, orderbook = read_paired(FIXTURES / "tiny_message.csv", FIXTURES / "tiny_orderbook.csv", 1)
    assert verify_snapshots(to_events(messages), depths(orderbook, 1), OrderBook()) == []


def test_lobster_window_that_opens_with_resting_orders_replays_exactly() -> None:
    messages, orderbook = read_paired(FIXTURES / "seeded_message.csv", FIXTURES / "seeded_orderbook.csv", 2)
    snapshots = depths(orderbook, 2)
    events = to_events(messages, first_row=1)
    assert verify_snapshots(events, snapshots[1:], seed_book(snapshots[0])) == []


def test_snapshot_mismatches_are_reported_by_row() -> None:
    messages, orderbook = read_paired(FIXTURES / "tiny_message.csv", FIXTURES / "tiny_orderbook.csv", 1)
    snapshots = depths(orderbook, 1)
    snapshots[2] = ([(585000, 999)], snapshots[2][1])
    assert verify_snapshots(to_events(messages), snapshots, OrderBook()) == [2]

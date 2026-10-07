"""Phase 3: classifying order events into the six-type flow alphabet."""

import numpy as np
import pytest

from microstructure.book import Side
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
from microstructure.flow import ClassifiedFlow, FlowMarks, FlowType, classify, marks

TICK = 100  # $0.01 in ITCH/LOBSTER price units


def _book() -> list[OrderEvent]:
    """Bids 1000x10 (id 1), 999x10 (id 2); asks 1002x10 (ids 3, 4), 1003x10 (id 5)."""
    return [
        AddOrder(ts=1, order_id=1, side=Side.BID, price=1000 * TICK, qty=10),
        AddOrder(ts=1, order_id=2, side=Side.BID, price=999 * TICK, qty=10),
        AddOrder(ts=1, order_id=3, side=Side.ASK, price=1002 * TICK, qty=10),
        AddOrder(ts=1, order_id=4, side=Side.ASK, price=1002 * TICK, qty=10),
        AddOrder(ts=1, order_id=5, side=Side.ASK, price=1003 * TICK, qty=10),
    ]


def _tail(flow: ClassifiedFlow, n: int) -> list[tuple[FlowType, int, int]]:
    """Last n classified events as (type, qty, distance)."""
    return [
        (FlowType(k), int(q), int(d))
        for k, q, d in zip(flow.kind[-n:], flow.qty[-n:], flow.distance[-n:], strict=True)
    ]


def test_a_sweep_across_levels_is_one_market_order() -> None:
    sweep = [ExecuteOrder(ts=5, order_id=i, qty=q) for i, q in ((3, 10), (4, 10), (5, 4))]
    flow = classify([*_book(), *sweep], tick=TICK)
    assert _tail(flow, 1) == [(FlowType.MB, 24, 0)]
    assert len(flow) == 6  # five adds + one market buy


def test_executions_at_different_times_are_separate_market_orders() -> None:
    events = [*_book(), ExecuteOrder(ts=5, order_id=3, qty=2), ExecuteOrder(ts=6, order_id=3, qty=3)]
    assert _tail(classify(events, tick=TICK), 2) == [(FlowType.MB, 2, 0), (FlowType.MB, 3, 0)]


def test_execution_of_a_resting_bid_is_a_market_sell() -> None:
    assert _tail(classify([*_book(), ExecuteOrder(ts=5, order_id=1, qty=4)], tick=TICK), 1) == [
        (FlowType.MS, 4, 0)
    ]


def test_limit_adds_are_split_by_position_relative_to_the_best() -> None:
    events = [
        *_book(),
        AddOrder(ts=5, order_id=10, side=Side.BID, price=1000 * TICK, qty=7),  # at best
        AddOrder(ts=6, order_id=11, side=Side.BID, price=1001 * TICK, qty=8),  # inside, 1 tick better
        AddOrder(ts=7, order_id=12, side=Side.ASK, price=1005 * TICK, qty=9),  # 3 ticks behind best ask 1002
    ]
    assert _tail(classify(events, tick=TICK), 3) == [
        (FlowType.LA, 7, 0),
        (FlowType.LI, 8, 1),
        (FlowType.LD, 9, 3),
    ]


def test_first_order_on_an_empty_side_counts_as_at_the_best() -> None:
    flow = classify([AddOrder(ts=1, order_id=1, side=Side.ASK, price=1000 * TICK, qty=5)], tick=TICK)
    assert _tail(flow, 1) == [(FlowType.LA, 5, 0)]


def test_cancels_record_their_distance_from_the_best() -> None:
    events = [*_book(), CancelOrder(ts=5, order_id=2, qty=4), DeleteOrder(ts=6, order_id=5)]
    assert _tail(classify(events, tick=TICK), 2) == [(FlowType.C, 4, 1), (FlowType.C, 10, 1)]


def test_replace_is_a_cancel_then_an_add_classified_after_the_cancel() -> None:
    # Order 1 is alone at the best bid; moving it down one tick leaves 999 as
    # the best, so the new half is an add *at* the best.
    events = [*_book(), ReplaceOrder(ts=5, order_id=1, new_order_id=9, price=999 * TICK, qty=10)]
    assert _tail(classify(events, tick=TICK), 2) == [(FlowType.C, 10, 0), (FlowType.LA, 10, 0)]


def test_hidden_trades_are_signed_by_price_relative_to_mid() -> None:
    mid = 1001 * TICK
    events = [
        *_book(),
        HiddenTrade(ts=5, resting_side=Side.BID, price=mid + 50, qty=3),
        HiddenTrade(ts=6, resting_side=Side.BID, price=mid - 50, qty=4),
        HiddenTrade(ts=7, resting_side=Side.BID, price=mid, qty=5),
    ]
    flow = classify(events, tick=TICK)
    assert _tail(flow, 2) == [(FlowType.MB, 3, 0), (FlowType.MS, 4, 0)]
    assert flow.unsigned_trades == 1


def test_cross_executions_are_not_market_orders_but_printable_ones_are() -> None:
    events = [
        *_book(),
        ExecuteOrderWithPrice(ts=5, order_id=1, qty=2, price=1000 * TICK, printable=False),
        ExecuteOrderWithPrice(ts=6, order_id=3, qty=2, price=1002 * TICK, printable=True),
    ]
    flow = classify(events, tick=TICK)
    assert _tail(flow, 1) == [(FlowType.MB, 2, 0)]
    assert len(flow) == 6


def test_unknown_order_references_are_skipped_and_reported() -> None:
    flow = classify([*_book(), DeleteOrder(ts=5, order_id=99)], tick=TICK)
    assert len(flow) == 5
    assert flow.replay.unknown_order_refs == 1


def test_between_selects_a_half_open_time_window() -> None:
    events = [*_book(), ExecuteOrder(ts=5, order_id=3, qty=1), ExecuteOrder(ts=9, order_id=3, qty=1)]
    window = classify(events, tick=TICK).between(2, 9)
    assert list(window.ts) == [5]


def test_to_stream_measures_seconds_from_the_window_start() -> None:
    events = [*_book(), ExecuteOrder(ts=1_500_000_000, order_id=1, qty=1)]
    stream = classify(events, tick=TICK).to_stream(1_000_000_000, 3_000_000_000)
    assert stream.horizon == pytest.approx(2.0)
    np.testing.assert_allclose(stream.times, [0.5])
    assert list(stream.types) == [FlowType.MS]


def test_marks_collect_sizes_and_distances_per_type() -> None:
    events = [
        *_book(),
        AddOrder(ts=5, order_id=12, side=Side.ASK, price=1005 * TICK, qty=9),
        CancelOrder(ts=6, order_id=2, qty=4),
    ]
    m = marks(classify(events, tick=TICK))
    # Orders 2 (999 behind 1000) and 5 (1003 behind 1002) from _book, then order 12.
    assert list(m.sizes[FlowType.LD]) == [10, 10, 9]
    assert list(m.distances[FlowType.LD]) == [1, 1, 3]
    assert list(m.distances[FlowType.C]) == [1]


def test_marks_within_a_distance_drop_far_placements_only() -> None:
    m = FlowMarks(
        sizes={kind: np.array([100, 5_000]) for kind in FlowType},
        distances={
            FlowType.LI: np.array([1, 2]),
            FlowType.LD: np.array([1, 3, 795, 2_660_264]),
            FlowType.C: np.array([0, 4, 900]),
        },
    )
    near = m.within(4)
    assert list(near.distances[FlowType.LD]) == [1, 3]
    assert list(near.distances[FlowType.C]) == [0, 4]
    assert list(near.distances[FlowType.LI]) == [1, 2]
    assert list(near.sizes[FlowType.MB]) == [100, 5_000]  # sizes are untouched

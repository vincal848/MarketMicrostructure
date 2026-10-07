"""Order book invariants: price-time priority, FIFO fills, exact cancels,
and validation that rejects bad input before any state changes."""

import contextlib
import random

import pytest

from microstructure.book import Fill, OrderBook, Side


def _ids_and_qtys(fills: list[Fill]) -> list[tuple[int, int]]:
    return [(f.resting_order_id, f.qty) for f in fills]


def test_best_bid_stays_below_best_ask_after_any_sequence_of_crossing_orders() -> None:
    book = OrderBook()
    rng = random.Random(0)
    for order_id in range(500):
        side = rng.choice([Side.BID, Side.ASK])
        book.add_limit_order(order_id, side, rng.randint(95, 105), rng.randint(1, 20))
        best_bid, best_ask = book.best_bid(), book.best_ask()
        if best_bid is not None and best_ask is not None:
            assert best_bid < best_ask


def test_book_matches_a_brute_force_reference_under_random_flow() -> None:
    # Differential test: after every operation, the sorted-level book must
    # agree with depth recomputed directly from the set of live orders.
    book = OrderBook()
    rng = random.Random(1)
    live: dict[int, tuple[Side, int]] = {}
    for order_id in range(2000):
        action = rng.random()
        if action < 0.6:
            side = rng.choice([Side.BID, Side.ASK])
            price = rng.randint(90, 110)
            book.add_limit_order(order_id, side, price, rng.randint(1, 30))
            live[order_id] = (side, price)
        elif action < 0.8 and live:
            victim = rng.choice(sorted(live))
            with contextlib.suppress(KeyError):  # may already be filled by a crossing order
                book.cancel_order(victim)
            live.pop(victim)
        else:
            book.market_order(rng.choice([Side.BID, Side.ASK]), rng.randint(1, 50))

        resting = book._orders.values()
        for side, levels in zip((Side.BID, Side.ASK), book.depth_snapshot(1000), strict=True):
            expected: dict[int, int] = {}
            for order in resting:
                if order.side is side:
                    expected[order.price] = expected.get(order.price, 0) + order.qty
            best_first = sorted(expected.items(), reverse=side is Side.BID)
            assert levels == best_first


def test_market_order_consumes_resting_volume_fifo_at_a_level() -> None:
    book = OrderBook()
    book.add_limit_order(1, Side.ASK, 100, 10)
    book.add_limit_order(2, Side.ASK, 100, 10)
    fills, leftover = book.market_order(Side.BID, 15)
    assert leftover == 0
    assert _ids_and_qtys(fills) == [(1, 10), (2, 5)]
    assert book.depth_snapshot(5)[1] == [(100, 5)]


def test_market_order_walks_levels_best_price_first() -> None:
    book = OrderBook()
    book.add_limit_order(1, Side.BID, 99, 5)
    book.add_limit_order(2, Side.BID, 100, 5)
    fills, _ = book.market_order(Side.ASK, 8)
    assert [(f.price, f.qty) for f in fills] == [(100, 5), (99, 3)]


def test_market_order_reports_leftover_when_the_book_runs_dry() -> None:
    book = OrderBook()
    book.add_limit_order(1, Side.ASK, 100, 5)
    fills, leftover = book.market_order(Side.BID, 20)
    assert leftover == 15
    assert _ids_and_qtys(fills) == [(1, 5)]
    assert book.best_ask() is None


def test_cancel_removes_exactly_the_requested_volume() -> None:
    book = OrderBook()
    book.add_limit_order(1, Side.BID, 100, 20)
    book.cancel_order(1, qty=8)
    assert book.depth_snapshot(5)[0] == [(100, 12)]
    book.cancel_order(1)
    assert book.depth_snapshot(5)[0] == []
    with pytest.raises(KeyError):
        book.cancel_order(1)


def test_cancelling_a_non_best_level_keeps_the_best() -> None:
    book = OrderBook()
    book.add_limit_order(1, Side.ASK, 101, 5)
    book.add_limit_order(2, Side.ASK, 103, 5)
    book.cancel_order(2)
    assert book.best_ask() == 101
    assert book.depth_snapshot(5)[1] == [(101, 5)]


def test_depth_snapshot_reports_top_n_levels_best_first() -> None:
    book = OrderBook()
    for i, price in enumerate([99, 98, 100, 97]):
        book.add_limit_order(i, Side.BID, price, 10)
    for i, price in enumerate([101, 103, 102]):
        book.add_limit_order(10 + i, Side.ASK, price, 10)
    assert book.depth_snapshot(2) == ([(100, 10), (99, 10)], [(101, 10), (102, 10)])
    assert book.depth_snapshot(0) == ([], [])


def test_crossing_limit_order_fills_before_resting_its_remainder() -> None:
    book = OrderBook()
    book.add_limit_order(1, Side.ASK, 100, 10)
    fills = book.add_limit_order(2, Side.BID, 101, 15)
    assert _ids_and_qtys(fills) == [(1, 10)]
    assert book.depth_snapshot(5) == ([(101, 5)], [])


def test_duplicate_order_id_is_rejected() -> None:
    book = OrderBook()
    book.add_limit_order(1, Side.BID, 100, 10)
    with pytest.raises(ValueError, match="duplicate"):
        book.add_limit_order(1, Side.BID, 99, 5)


def test_non_positive_quantity_is_rejected() -> None:
    book = OrderBook()
    with pytest.raises(ValueError, match="qty"):
        book.add_limit_order(1, Side.BID, 100, 0)
    with pytest.raises(ValueError, match="qty"):
        book.market_order(Side.BID, -5)
    book.add_limit_order(2, Side.BID, 100, 10)
    with pytest.raises(ValueError, match="qty"):
        book.cancel_order(2, qty=0)


def test_unknown_side_is_rejected_without_touching_the_book() -> None:
    # Regression: "buy" used to be silently treated as a sell, and a
    # typo'd side on a crossing limit order traded before raising.
    book = OrderBook()
    book.add_limit_order(1, Side.BID, 100, 10)
    with pytest.raises(ValueError, match="buy"):
        book.market_order("buy", 5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ASK"):
        book.add_limit_order(2, "ASK", 99, 5)  # type: ignore[arg-type]
    assert book.depth_snapshot(5) == ([(100, 10)], [])


def test_fractional_price_is_rejected() -> None:
    book = OrderBook()
    with pytest.raises(TypeError):
        book.add_limit_order(1, Side.BID, 100.5, 10)  # type: ignore[arg-type]

"""Order book invariants: price-time priority, FIFO fills, exact cancels."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import pytest

from lob import OrderBook


def test_best_bid_stays_below_best_ask_after_any_sequence_of_crossing_orders():
    book = OrderBook()
    rng = random.Random(0)
    next_id = 0
    for _ in range(500):
        side = rng.choice(["bid", "ask"])
        price = rng.randint(95, 105)
        qty = rng.randint(1, 20)
        book.add_limit_order(next_id, side, price, qty)
        next_id += 1
        bb, ba = book.best_bid(), book.best_ask()
        if bb is not None and ba is not None:
            assert bb < ba


def test_market_order_consumes_resting_volume_fifo_at_a_level():
    book = OrderBook()
    book.add_limit_order(1, "ask", 100, 10)
    book.add_limit_order(2, "ask", 100, 10)
    fills, leftover = book.market_order("bid", 15)
    assert leftover == 0
    assert [(f.resting_order_id, f.qty) for f in fills] == [(1, 10), (2, 5)]
    _, asks = book.depth_snapshot(5)
    assert asks == [(100, 5)]


def test_market_order_reports_leftover_when_the_book_runs_dry():
    book = OrderBook()
    book.add_limit_order(1, "ask", 100, 5)
    fills, leftover = book.market_order("bid", 20)
    assert leftover == 15
    assert [(f.resting_order_id, f.qty) for f in fills] == [(1, 5)]
    assert book.best_ask() is None


def test_cancel_removes_exactly_the_requested_volume():
    book = OrderBook()
    book.add_limit_order(1, "bid", 100, 20)
    book.cancel_order(1, qty=8)
    bids, _ = book.depth_snapshot(5)
    assert bids == [(100, 12)]
    book.cancel_order(1)
    bids, _ = book.depth_snapshot(5)
    assert bids == []
    with pytest.raises(KeyError):
        book.cancel_order(1)


def test_depth_snapshot_reports_top_n_levels_best_first():
    book = OrderBook()
    for i, price in enumerate([99, 98, 100, 97]):
        book.add_limit_order(i, "bid", price, 10)
    for i, price in enumerate([101, 103, 102]):
        book.add_limit_order(10 + i, "ask", price, 10)
    bids, asks = book.depth_snapshot(2)
    assert bids == [(100, 10), (99, 10)]
    assert asks == [(101, 10), (102, 10)]


def test_crossing_limit_order_fills_before_resting_its_remainder():
    book = OrderBook()
    book.add_limit_order(1, "ask", 100, 10)
    fills = book.add_limit_order(2, "bid", 101, 15)
    assert [(f.resting_order_id, f.qty) for f in fills] == [(1, 10)]
    bids, asks = book.depth_snapshot(5)
    assert bids == [(101, 5)]
    assert asks == []


def test_duplicate_order_id_is_rejected():
    book = OrderBook()
    book.add_limit_order(1, "bid", 100, 10)
    with pytest.raises(ValueError):
        book.add_limit_order(1, "bid", 99, 5)


def test_non_positive_quantity_is_rejected():
    book = OrderBook()
    with pytest.raises(ValueError):
        book.add_limit_order(1, "bid", 100, 0)
    with pytest.raises(ValueError):
        book.market_order("bid", -5)

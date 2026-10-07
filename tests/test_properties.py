"""Phase 7: property-based tests of the order book under arbitrary flow."""

from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st

from microstructure.book import OrderBook, Side


@dataclass(frozen=True)
class Limit:
    side: Side
    price: int
    qty: int


@dataclass(frozen=True)
class Market:
    side: Side
    qty: int


@dataclass(frozen=True)
class Cancel:
    order_id: int
    qty: int


@dataclass(frozen=True)
class Execute:
    order_id: int
    qty: int


Operation = Limit | Market | Cancel | Execute

sides = st.sampled_from([Side.BID, Side.ASK])
ids = st.integers(0, 199)
operations = st.lists(
    st.one_of(
        st.builds(Limit, sides, st.integers(90, 110), st.integers(1, 50)),
        st.builds(Market, sides, st.integers(1, 120)),
        st.builds(Cancel, ids, st.integers(1, 60)),
        st.builds(Execute, ids, st.integers(1, 60)),
    ),
    max_size=200,
)


@settings(max_examples=300, deadline=None)
@given(operations)
def test_book_invariants_hold_under_any_sequence_of_operations(ops: list[Operation]) -> None:
    book = OrderBook()
    expected_resting = 0
    for next_id, op in enumerate(ops):
        match op:
            case Limit(side, price, qty):
                filled = sum(f.qty for f in book.add_limit_order(next_id, side, price, qty))
                expected_resting += qty - 2 * filled  # its own filled part never rests, and consumed as much
            case Market(side, qty):
                fills, _ = book.market_order(side, qty)
                expected_resting -= sum(f.qty for f in fills)
            case Cancel(order_id, qty) if order_id in book:
                removed = min(qty, book.resting_order(order_id).qty)
                book.cancel_order(order_id, qty)
                expected_resting -= removed
            case Execute(order_id, qty) if order_id in book:
                executed = min(qty, book.resting_order(order_id).qty)
                book.execute_order(order_id, executed)
                expected_resting -= executed

        bid, ask = book.best_bid(), book.best_ask()
        assert bid is None or ask is None or bid < ask
        bids, asks = book.depth_snapshot(1_000)
        assert [p for p, _ in bids] == sorted((p for p, _ in bids), reverse=True)
        assert [p for p, _ in asks] == sorted(p for p, _ in asks)
        assert all(q > 0 for _, q in bids + asks)
        assert sum(q for _, q in bids + asks) == expected_resting


@settings(max_examples=200, deadline=None)
@given(operations)
def test_queue_position_is_the_volume_ahead_at_the_same_price(ops: list[Operation]) -> None:
    book = OrderBook()
    for next_id, op in enumerate(ops):
        if isinstance(op, Limit):
            book.add_limit_order(next_id, op.side, op.price, op.qty)
    bids, asks = book.depth_snapshot(1_000)
    for order_id in range(len(ops)):
        if order_id in book:
            order = book.resting_order(order_id)
            level = dict(bids if order.side is Side.BID else asks)[order.price]
            assert 0 <= book.queue_ahead(order_id) <= level - order.qty

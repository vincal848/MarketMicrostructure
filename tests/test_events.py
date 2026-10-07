"""Normalized order-event model shared by every data source."""

import pytest

from microstructure.book import Side
from microstructure.events import AddOrder, CancelOrder, ExecuteOrder, ReplaceOrder


def test_events_are_immutable_values() -> None:
    event = AddOrder(ts=1, order_id=7, side=Side.BID, price=1_000_000, qty=100)
    assert event == AddOrder(ts=1, order_id=7, side=Side.BID, price=1_000_000, qty=100)
    with pytest.raises(AttributeError):
        event.qty = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    "build",
    [
        lambda: AddOrder(ts=1, order_id=1, side=Side.BID, price=100, qty=0),
        lambda: CancelOrder(ts=1, order_id=1, qty=-1),
        lambda: ExecuteOrder(ts=1, order_id=1, qty=0),
        lambda: ReplaceOrder(ts=1, order_id=1, new_order_id=2, price=100, qty=0),
    ],
)
def test_non_positive_quantities_are_rejected(build: object) -> None:
    with pytest.raises(ValueError, match="qty"):
        build()  # type: ignore[operator]


def test_negative_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="ts"):
        AddOrder(ts=-1, order_id=1, side=Side.ASK, price=100, qty=1)

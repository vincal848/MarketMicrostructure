"""Phase 4: stylized facts computed identically on real and simulated tapes."""

import numpy as np
import pytest
from microstructure.stylized import (
    MarketTape,
    record_tape,
    signature_plot,
    spread_distribution,
    total_variation,
    trade_sign_autocorrelation,
)

from microstructure.book import Side
from microstructure.events import AddOrder, ExecuteOrder, HiddenTrade, OrderEvent

TICK = 100


def _tape(bid: list[int], ask: list[int], times: list[float] | None = None) -> MarketTape:
    n = len(bid)
    return MarketTape(
        times=np.arange(n, dtype=float) if times is None else np.array(times),
        bid=np.array(bid),
        ask=np.array(ask),
        trade_times=np.array([]),
        trade_prices=np.array([], dtype=np.int64),
        trade_qty=np.array([], dtype=np.int64),
        trade_sign=np.array([], dtype=np.int64),
        horizon=float(n),
    )


def test_spread_distribution_buckets_ticks_and_caps_the_tail() -> None:
    tape = _tape(bid=[100, 100, 100, 100], ask=[200, 200, 300, 900])
    np.testing.assert_allclose(spread_distribution(tape, TICK, max_ticks=3), [0.5, 0.25, 0.25])


def test_alternating_trade_signs_have_lag_one_autocorrelation_of_minus_one() -> None:
    signs = np.array([1, -1] * 50)
    acf = trade_sign_autocorrelation(signs, max_lag=2)
    assert acf[0] == pytest.approx(-1.0, abs=0.02)
    assert acf[1] == pytest.approx(1.0, abs=0.02)


def test_signature_plot_of_a_known_path() -> None:
    # Mid steps +1 tick at t = 0.5, 1.5, 2.5, 3.5 (horizon 4): sampled every
    # second there are 4 moves of 1 tick -> RV = 4 * 100^2 / 4 s; every
    # 2 seconds, 2 moves of 2 ticks -> RV = 2 * 200^2 / 4 s.
    times = [0.0, 0.5, 1.5, 2.5, 3.5]
    bid = [1000, 1100, 1200, 1300, 1400]
    ask = [b + 200 for b in bid]
    tape = _tape(bid, ask, times)
    tape = MarketTape(**{**tape.__dict__, "horizon": 4.0})
    np.testing.assert_allclose(signature_plot(tape, intervals=[1.0, 2.0]), [10_000.0, 20_000.0])


def test_total_variation_distance() -> None:
    assert total_variation(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0
    assert total_variation(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0


def test_record_tape_from_replayed_events() -> None:
    second = 1_000_000_000
    events: list[OrderEvent] = [
        AddOrder(ts=1 * second, order_id=1, side=Side.BID, price=1000 * TICK, qty=10),
        AddOrder(ts=2 * second, order_id=2, side=Side.ASK, price=1002 * TICK, qty=10),
        ExecuteOrder(ts=3 * second, order_id=2, qty=4),
        HiddenTrade(ts=4 * second, resting_side=Side.BID, price=1000 * TICK + 50, qty=5),
    ]
    tape = record_tape(events, start_ns=0, end_ns=10 * second)
    # A quote is recorded only once both sides exist.
    np.testing.assert_allclose(tape.times, [2.0, 3.0, 4.0])
    assert list(tape.bid) == [1000 * TICK] * 3
    np.testing.assert_allclose(tape.trade_times, [3.0, 4.0])
    assert list(tape.trade_sign) == [1, -1]  # buy hit the ask; hidden trade below mid
    assert list(tape.trade_qty) == [4, 5]
    assert tape.horizon == 10.0

"""Phase 5: PnL ledger and attribution."""

import numpy as np
import pytest

from microstructure.accounting import Ledger, MidPath
from microstructure.book import Side
from microstructure.simulator import AgentFill


def _fill(time: float, side: Side, price: int, qty: int, mid: float, aggressive: bool = False) -> AgentFill:
    return AgentFill(time=time, side=side, price=price, qty=qty, aggressive=aggressive, mid=mid)


def test_cash_inventory_and_mark_to_market() -> None:
    ledger = Ledger()
    ledger.record(_fill(1.0, Side.BID, 999, 100, 1000.0))
    ledger.record(_fill(2.0, Side.ASK, 1002, 40, 1001.0))
    assert ledger.inventory == 60
    assert ledger.cash == -999 * 100 + 1002 * 40
    assert ledger.mark_to_market(1005.0) == pytest.approx(-99_900 + 40_080 + 60 * 1005.0)


def test_fees_charge_takers_and_rebate_makers() -> None:
    ledger = Ledger(maker_fee=-0.2, taker_fee=0.3)  # per share, price units
    ledger.record(_fill(1.0, Side.BID, 1000, 100, 1000.5, aggressive=False))
    ledger.record(_fill(2.0, Side.ASK, 1000, 50, 1000.5, aggressive=True))
    assert ledger.fees == pytest.approx(-0.2 * 100 + 0.3 * 50)
    assert ledger.mark_to_market(1000.0) == pytest.approx(-1000 * 100 + 1000 * 50 + 50 * 1000.0 - ledger.fees)


def test_hand_computed_attribution() -> None:
    # Buy 100 at 999 when mid = 1000; one second later mid = 998; at the end 1003.
    # Sell 100 at 1004 when mid = 1003; one second later mid = 1005; end 1003.
    path = MidPath(
        times=np.array([0.0, 1.0, 2.0, 4.0, 5.0]), mids=np.array([1000.0, 1000.0, 998.0, 1003.0, 1005.0])
    )
    ledger = Ledger()
    ledger.record(_fill(1.0, Side.BID, 999, 100, 1000.0))
    ledger.record(_fill(4.0, Side.ASK, 1004, 100, 1003.0))
    pnl = ledger.attribution(path, horizon=1.0, final_mid=1003.0)
    assert pnl.spread_capture == pytest.approx(100 * 1 + 100 * 1)
    assert pnl.adverse_selection == pytest.approx(100 * (998 - 1000) + -100 * (1005 - 1003))
    assert pnl.inventory == pytest.approx(100 * (1003 - 998) + -100 * (1003 - 1005))
    assert pnl.total == pytest.approx(ledger.mark_to_market(1003.0))


def test_components_sum_to_mark_to_market_on_random_fills() -> None:
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(0, 100, 400))
    mids = 10_000 + np.cumsum(rng.normal(0, 3, 400))
    path = MidPath(times=times, mids=mids)
    ledger = Ledger(maker_fee=-0.1, taker_fee=0.25)
    for k in rng.choice(400, 60, replace=False):
        side = Side.BID if rng.uniform() < 0.5 else Side.ASK
        offset = int(rng.integers(-3, 4))
        ledger.record(
            _fill(
                float(times[k]),
                side,
                int(round(mids[k])) + offset,
                int(rng.integers(1, 300)),
                float(mids[k]),
                bool(rng.uniform() < 0.3),
            )
        )
    pnl = ledger.attribution(path, horizon=5.0, final_mid=float(mids[-1]))
    assert pnl.spread_capture + pnl.adverse_selection + pnl.inventory - pnl.fees == pytest.approx(pnl.total)
    assert pnl.total == pytest.approx(ledger.mark_to_market(float(mids[-1])))


def test_mid_path_is_the_last_mid_at_or_before_a_time() -> None:
    path = MidPath(times=np.array([1.0, 2.0, 3.0]), mids=np.array([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(path.at(np.array([0.5, 1.0, 2.5, 9.0])), [10.0, 10.0, 20.0, 30.0])


def test_inventory_excursion_statistics() -> None:
    ledger = Ledger()
    for side, qty in ((Side.BID, 100), (Side.BID, 50), (Side.ASK, 300), (Side.BID, 100)):
        ledger.record(_fill(0.0, side, 1000, qty, 1000.0))
    assert ledger.max_abs_inventory == 150
    assert ledger.inventory == -50
    assert ledger.volume == 550

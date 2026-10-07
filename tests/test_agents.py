"""Phase 5: market-making agents and the parameter estimators they need."""

import math
from collections.abc import Sequence

import numpy as np
import pytest

from microstructure.agents import (
    AvellanedaStoikovAgent,
    FixedSpreadAgent,
    estimate_fill_curve,
    estimate_sigma,
)
from microstructure.avellaneda_stoikov import ASParams, quotes
from microstructure.book import Side
from microstructure.simulator import AgentFill, MarketView, Quote
from microstructure.stylized import MarketTape

TICK = 100


def _view(bid: int, ask: int, time: float = 0.0) -> MarketView:
    return MarketView(
        time=time, best_bid=bid, best_ask=ask, depth=([(bid, 100)], [(ask, 100)]), own_orders=()
    )


def _fill(side: Side, qty: int) -> AgentFill:
    return AgentFill(time=0.0, side=side, price=1_000_000, qty=qty, aggressive=False, mid=1_000_050.0)


def _only_quote(actions: Sequence[object]) -> Quote:
    assert len(actions) == 1
    assert isinstance(actions[0], Quote)
    return actions[0]


def _price(level: tuple[int, int] | None) -> int:
    assert level is not None
    return level[0]


def _size(level: tuple[int, int] | None) -> int:
    assert level is not None
    return level[1]


AS_TICKS = ASParams(gamma=0.01, sigma=2.0, kappa=0.3)  # in tick units


def test_as_quotes_match_the_closed_form_in_ticks_rounded_outward() -> None:
    agent = AvellanedaStoikovAgent(AS_TICKS, tick=TICK, size=100, horizon=60.0)
    quote = _only_quote(
        agent.decide(_view(999_000, 1_001_000, time=0.0))
    )  # mid = 10_000 ticks, spread 20 ticks
    exact = quotes(10_000.0, inventory=0, time_to_go=60.0, params=AS_TICKS)
    assert quote.bid == (math.floor(exact.bid) * TICK, 100)
    assert quote.ask == (math.ceil(exact.ask) * TICK, 100)


def test_long_inventory_lowers_both_quotes() -> None:
    flat = AvellanedaStoikovAgent(AS_TICKS, tick=TICK, size=100, horizon=60.0)
    long = AvellanedaStoikovAgent(AS_TICKS, tick=TICK, size=100, horizon=60.0)
    long.on_fill(_fill(Side.BID, 300))
    view = _view(999_000, 1_001_000)
    flat_quote, long_quote = _only_quote(flat.decide(view)), _only_quote(long.decide(view))
    assert _price(long_quote.bid) < _price(flat_quote.bid)
    assert _price(long_quote.ask) < _price(flat_quote.ask)


def test_quotes_never_cross_the_book() -> None:
    # A huge inventory skew would put the ask below the best bid; it is
    # clipped to post-only prices instead.
    agent = AvellanedaStoikovAgent(
        ASParams(gamma=1.0, sigma=5.0, kappa=0.3), tick=TICK, size=100, horizon=60.0
    )
    agent.on_fill(_fill(Side.BID, 5_000))
    quote = _only_quote(agent.decide(_view(1_000_000, 1_000_100)))
    assert _price(quote.ask) >= 1_000_100
    assert _price(quote.bid) <= 1_000_000


def test_inventory_cap_limits_the_quoted_size() -> None:
    agent = AvellanedaStoikovAgent(AS_TICKS, tick=TICK, size=100, horizon=60.0, inventory_cap=250)
    agent.on_fill(_fill(Side.BID, 200))
    quote = _only_quote(agent.decide(_view(999_000, 1_001_000)))
    assert _size(quote.bid) == 50  # only 50 more fits under the cap
    agent.on_fill(_fill(Side.BID, 50))
    quote = _only_quote(agent.decide(_view(999_000, 1_001_000)))
    assert quote.bid is None
    assert _size(quote.ask) == 100


def test_inventory_tracks_fills() -> None:
    agent = AvellanedaStoikovAgent(AS_TICKS, tick=TICK, size=100, horizon=60.0)
    agent.on_fill(_fill(Side.BID, 120))
    agent.on_fill(_fill(Side.ASK, 20))
    assert agent.inventory == 100


def test_fixed_spread_agent_quotes_symmetrically_around_mid() -> None:
    agent = FixedSpreadAgent(half_spread_ticks=3, tick=TICK, size=200)
    quote = _only_quote(agent.decide(_view(999_900, 1_000_100)))  # mid 1_000_000
    assert quote.bid == (999_700, 200)
    assert quote.ask == (1_000_300, 200)


def test_agents_stand_aside_on_a_one_sided_book() -> None:
    agent = FixedSpreadAgent(half_spread_ticks=3, tick=TICK, size=200)
    view = MarketView(
        time=0.0, best_bid=None, best_ask=1_000_000, depth=([], [(1_000_000, 5)]), own_orders=()
    )
    assert agent.decide(view) == []


def _tape_from_market_orders(depths_in_ticks: np.ndarray, horizon: float) -> MarketTape:
    """One fill per market order, `depth` ticks from a constant mid."""
    n = len(depths_in_ticks)
    times = np.linspace(0.0, horizon, n, endpoint=False)
    mid = 1_000_000
    return MarketTape(
        times=times,
        bid=np.full(n, mid - TICK // 2),
        ask=np.full(n, mid + TICK // 2),
        trade_times=times,
        trade_prices=(mid + TICK // 2 + depths_in_ticks * TICK).astype(np.int64),
        trade_qty=np.full(n, 100),
        trade_sign=np.ones(n, dtype=np.int64),
        horizon=horizon,
    )


def test_fill_curve_estimate_recovers_an_exponential_decay() -> None:
    rng = np.random.default_rng(0)
    kappa = 0.7
    depths = np.floor(rng.exponential(1 / kappa, 20_000))  # ticks beyond the touch
    a, k = estimate_fill_curve(_tape_from_market_orders(depths, horizon=1000.0), tick=TICK, max_ticks=8)
    assert k == pytest.approx(kappa, rel=0.05)
    assert a == pytest.approx(20.0, rel=0.1)  # 20,000 orders in 1000 s reach the touch


def test_sigma_estimate_of_a_random_walk() -> None:
    rng = np.random.default_rng(1)
    steps = rng.choice([-TICK, TICK], size=36_000)  # one tick every 0.1 s -> variance 10 ticks^2 / s
    mid = 1_000_000 + np.cumsum(steps)
    tape = MarketTape(
        times=np.arange(36_000, dtype=np.float64) * 0.1,
        bid=mid - TICK,
        ask=mid + TICK,
        trade_times=np.array([]),
        trade_prices=np.array([], dtype=np.int64),
        trade_qty=np.array([], dtype=np.int64),
        trade_sign=np.array([], dtype=np.int64),
        horizon=3600.0,
    )
    assert estimate_sigma(tape, tick=TICK, interval=10.0) == pytest.approx(math.sqrt(10.0), rel=0.1)

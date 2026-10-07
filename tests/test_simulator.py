"""Phase 4: the discrete-event market simulator."""

from dataclasses import dataclass, field

import numpy as np
import pytest
from microstructure.simulator import (
    Agent,
    AgentFill,
    MarketSimulator,
    MarketView,
    Quote,
    SendMarketOrder,
    SimulationConfig,
)

from microstructure.book import Side
from microstructure.flow import FlowMarks, FlowType
from microstructure.hawkes import HawkesParams

TICK = 100
MID = 1_000_000  # $100.00 in 1/10000 dollars


def _params() -> HawkesParams:
    alpha = np.full((6, 6), 0.02)
    np.fill_diagonal(alpha, 0.3)
    mu = [0.5, 0.5, 1.5, 0.3, 1.2, 2.0]  # MB MS LA LI LD C
    return HawkesParams(mu=mu, alpha=alpha, beta=np.full((6, 6), 2.0))


def _marks() -> FlowMarks:
    sizes = {kind: np.array([100, 200, 300]) for kind in FlowType}
    distances = {
        FlowType.LI: np.array([1]),
        FlowType.LD: np.array([1, 2, 3, 5]),
        FlowType.C: np.array([0, 1, 2]),
    }
    return FlowMarks(sizes, distances)


def _depth() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    bids = [(MID - TICK * (i + 1), 500) for i in range(10)]
    asks = [(MID + TICK * i, 500) for i in range(10)]
    return bids, asks


def _simulator(
    horizon: float = 300.0, seed: int = 0, agents: tuple[Agent, ...] = (), latency: float = 0.0
) -> MarketSimulator:
    config = SimulationConfig(horizon=horizon, tick=TICK, seed=seed, latency=latency)
    return MarketSimulator(_params(), _marks(), _depth(), config, agents=agents)


@dataclass
class FarQuoter:
    """Quotes 50 ticks away from the touch once, then never again."""

    decision_interval: float = 1e9
    fills: list[AgentFill] = field(default_factory=list)

    def decide(self, view: MarketView) -> list[Quote | SendMarketOrder]:
        return [Quote(bid=(MID - 50 * TICK, 7), ask=(MID + 50 * TICK, 9))]

    def on_fill(self, fill: AgentFill) -> None:
        self.fills.append(fill)


@dataclass
class TouchQuoter:
    """Joins the best bid and ask every second."""

    decision_interval: float = 1.0
    fills: list[AgentFill] = field(default_factory=list)
    views: list[MarketView] = field(default_factory=list)

    def decide(self, view: MarketView) -> list[Quote | SendMarketOrder]:
        self.views.append(view)
        if view.best_bid is None or view.best_ask is None:
            return []
        return [Quote(bid=(view.best_bid, 100), ask=(view.best_ask, 100))]

    def on_fill(self, fill: AgentFill) -> None:
        self.fills.append(fill)


@dataclass
class OneMarketBuy:
    decision_interval: float = 10.0
    sent: bool = False

    def decide(self, view: MarketView) -> list[Quote | SendMarketOrder]:
        if self.sent:
            return []
        self.sent = True
        return [SendMarketOrder(Side.BID, 100)]

    def on_fill(self, fill: AgentFill) -> None:
        pass


def test_background_event_rates_match_the_stationary_intensity() -> None:
    result = _simulator(horizon=4000.0).run()
    rates = result.generated / 4000.0
    assert rates == pytest.approx(_params().stationary_intensity(), rel=0.05)


def test_the_book_is_never_crossed() -> None:
    tape = _simulator(horizon=600.0).run().tape
    assert len(tape.times) > 1000
    assert np.all(tape.ask > tape.bid)


def test_the_same_seed_reproduces_the_run() -> None:
    first, second = _simulator(seed=7).run(), _simulator(seed=7).run()
    np.testing.assert_array_equal(first.tape.times, second.tape.times)
    np.testing.assert_array_equal(first.tape.bid, second.tape.bid)
    other = _simulator(seed=8).run()
    assert not np.array_equal(first.tape.times[:100], other.tape.times[:100])


def test_applied_flow_uses_the_calibrated_marks() -> None:
    flow = _simulator(horizon=600.0).run().flow
    ld = flow.distance[flow.kind == FlowType.LD]
    assert len(ld) > 100
    assert set(np.unique(ld)) <= {1, 2, 3, 5}
    assert set(np.unique(flow.qty[flow.kind == FlowType.LA])) <= {100, 200, 300}


def test_background_cancels_never_touch_agent_orders() -> None:
    agent = FarQuoter()
    simulator = _simulator(horizon=600.0, agents=(agent,))
    simulator.run()
    resting = simulator.agent_orders(0)
    assert sorted((o.side, o.qty) for o in resting) == [(Side.BID, 7), (Side.ASK, 9)]
    assert agent.fills == []


def test_agent_orders_reach_the_book_after_the_latency() -> None:
    simulator = _simulator(horizon=30.0, agents=(TouchQuoter(),), latency=0.25)
    result = simulator.run()
    assert result.order_log
    for sent, arrived in result.order_log:
        assert arrived - sent == pytest.approx(0.25)


def test_agent_fills_carry_side_price_and_quantity() -> None:
    agent = TouchQuoter()
    _simulator(horizon=300.0, agents=(agent,)).run()
    assert agent.fills
    for fill in agent.fills:
        assert fill.qty > 0
        assert fill.price % TICK == 0
        assert fill.side in (Side.BID, Side.ASK)
        assert not fill.aggressive


def test_unchanged_quotes_keep_their_queue_position() -> None:
    # Re-sending an identical quote every second must not cancel and re-add
    # the orders (which would send them to the back of the queue).
    agent = FarQuoter(decision_interval=1.0)
    simulator = _simulator(horizon=60.0, agents=(agent,))
    simulator.run()
    assert simulator.agent_order_count(0) == 2


def test_agent_market_orders_excite_the_background_flow() -> None:
    simulator = _simulator(horizon=1.0, agents=(OneMarketBuy(),))
    result = simulator.run()
    assert result.agent_market_orders == 1
    assert result.excitations[FlowType.MB] == 1


def test_agent_market_order_fills_are_aggressive() -> None:
    @dataclass
    class Recorder(OneMarketBuy):
        fills: list[AgentFill] = field(default_factory=list)

        def on_fill(self, fill: AgentFill) -> None:
            self.fills.append(fill)

    agent = Recorder()
    _simulator(horizon=1.0, agents=(agent,)).run()
    assert sum(f.qty for f in agent.fills) == 100
    assert all(f.aggressive and f.side is Side.BID for f in agent.fills)

"""Phase 5: paired, seeded evaluation of agents on identical simulated flow."""

from collections.abc import Callable

import numpy as np
import pytest

from microstructure.agents import AvellanedaStoikovAgent, FixedSpreadAgent
from microstructure.evaluation import Scenario, calibrate_avellaneda_stoikov, evaluate, paired_difference
from microstructure.flow import FlowMarks, FlowType
from microstructure.hawkes import HawkesParams
from microstructure.simulator import Agent

TICK = 100
MID = 1_000_000


def _scenario(horizon: float = 120.0) -> Scenario:
    alpha = np.full((6, 6), 0.02)
    np.fill_diagonal(alpha, 0.3)
    params = HawkesParams(mu=[0.6, 0.6, 1.5, 0.3, 1.2, 2.0], alpha=alpha, beta=np.full((6, 6), 2.0))
    marks = FlowMarks(
        sizes={kind: np.array([100, 200]) for kind in FlowType},
        distances={
            FlowType.LI: np.array([1]),
            FlowType.LD: np.array([1, 2, 3]),
            FlowType.C: np.array([0, 1, 2]),
        },
    )
    depth = ([(MID - TICK * (i + 1), 300) for i in range(10)], [(MID + TICK * i, 300) for i in range(10)])
    return Scenario(
        params=params, marks=marks, initial_depth=depth, tick=TICK, horizon=horizon, latency=0.001
    )


def test_paired_difference_of_identical_samples_is_exactly_zero() -> None:
    sample = [1.0, -2.0, 3.5, 0.25]
    result = paired_difference(sample, sample, seed=0)
    assert (result.mean, result.low, result.high) == (0.0, 0.0, 0.0)


def test_paired_difference_interval_covers_a_known_shift() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 200)
    result = paired_difference(list(base + 0.5 + rng.normal(0, 0.1, 200)), list(base), seed=1)
    assert result.low < 0.5 < result.high


def test_an_agent_evaluated_against_itself_differs_by_exactly_zero() -> None:
    make: dict[str, Callable[[], Agent]] = {
        "fixed": lambda: FixedSpreadAgent(half_spread_ticks=1, tick=TICK, size=100)
    }
    table = evaluate(_scenario(), make | {"fixed again": make["fixed"]}, seeds=range(3))
    diff = table.difference("fixed again", "fixed", metric="pnl")
    assert (diff.mean, diff.low, diff.high) == (0.0, 0.0, 0.0)


def test_avellaneda_stoikov_is_calibrated_from_the_simulator_itself() -> None:
    params = calibrate_avellaneda_stoikov(_scenario(), gamma=0.001, seeds=range(2), sigma_interval=5.0)
    assert params.gamma == 0.001
    assert 0.0 < params.sigma < 50.0  # ticks per sqrt(second)
    assert 0.0 < params.kappa < 10.0  # per tick
    # Calibrated in ticks, the quotes sit within a few ticks of the touch.
    assert (2 / params.gamma) * np.log1p(params.gamma / params.kappa) / 2 < 5.0


def test_baselines_trade_and_their_pnl_is_attributed() -> None:
    scenario = _scenario()
    as_params = calibrate_avellaneda_stoikov(scenario, gamma=0.001, seeds=range(100, 102), sigma_interval=5.0)
    agents: dict[str, Callable[[], Agent]] = {
        "fixed": lambda: FixedSpreadAgent(half_spread_ticks=1, tick=TICK, size=100),
        "as": lambda: AvellanedaStoikovAgent(as_params, tick=TICK, size=100, horizon=120.0),
    }
    table = evaluate(scenario, agents, seeds=range(4), attribution_horizon=1.0)
    for name in agents:
        runs = table.runs[name]
        assert len(runs) == 4
        assert all(run.fills > 0 for run in runs), name
        for run in runs:
            parts = run.spread_capture + run.adverse_selection + run.inventory_pnl - run.fees
            assert parts == pytest.approx(run.pnl)
    summary = table.summary()
    assert set(summary) == {"fixed", "as"}
    assert {"pnl", "fills", "spread_capture", "adverse_selection", "max_abs_inventory"} <= set(summary["as"])

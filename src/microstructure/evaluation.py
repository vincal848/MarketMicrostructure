"""Seeded, paired evaluation of market makers on identical simulated flow.

Each agent runs alone in its own simulation, once per seed, with the same
seeds for every agent (common random numbers). Pairing by seed removes
the luck of the draw from comparisons: the difference between two agents on
seed s is mostly due to the agents. The pairing is not exact, because an
agent's orders change the book and so which background orders later events
touch. A paired bootstrap of the per-seed differences gives the confidence
interval.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from microstructure.accounting import Ledger, MidPath
from microstructure.agents import estimate_fill_curve, estimate_sigma
from microstructure.avellaneda_stoikov import ASParams
from microstructure.book import Depth
from microstructure.flow import FlowMarks
from microstructure.hawkes import HawkesParams
from microstructure.simulator import Action, Agent, AgentFill, MarketSimulator, MarketView, SimulationConfig
from microstructure.stylized import MarketTape


@dataclass(frozen=True)
class Scenario:
    """Everything that defines the simulated market, minus the seed."""

    params: HawkesParams
    marks: FlowMarks
    initial_depth: Depth
    tick: int
    horizon: float
    latency: float = 0.0

    def config(self, seed: int) -> SimulationConfig:
        return SimulationConfig(horizon=self.horizon, tick=self.tick, seed=seed, latency=self.latency)


@dataclass(frozen=True)
class RunMetrics:
    seed: int
    pnl: float
    spread_capture: float
    adverse_selection: float
    inventory_pnl: float
    fees: float
    fills: int
    volume: int
    max_abs_inventory: int
    final_inventory: int


METRICS = tuple(f.name for f in fields(RunMetrics) if f.name != "seed")


@dataclass(frozen=True)
class Difference:
    """Mean of paired differences with a bootstrap confidence interval."""

    mean: float
    low: float
    high: float


def bootstrap_mean(
    values: Sequence[float], n_boot: int = 5000, level: float = 0.95, seed: int = 0
) -> Difference:
    """Mean of `values` with a percentile-bootstrap confidence interval."""
    sample = np.asarray(values, dtype=np.float64)
    if sample.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    means = sample[rng.integers(0, sample.size, size=(n_boot, sample.size))].mean(axis=1)
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return Difference(float(sample.mean()), float(low), float(high))


def paired_difference(
    a: Sequence[float], b: Sequence[float], n_boot: int = 5000, level: float = 0.95, seed: int = 0
) -> Difference:
    """Mean of a - b over pairs, with a percentile-bootstrap interval."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return bootstrap_mean(list(diff), n_boot, level, seed)


class _Recorded:
    """Forwards to an agent while booking its fills in a ledger."""

    def __init__(self, agent: Agent, ledger: Ledger) -> None:
        self.agent = agent
        self.ledger = ledger
        self.decision_interval = agent.decision_interval

    def decide(self, view: MarketView) -> list[Action]:
        return self.agent.decide(view)

    def on_fill(self, fill: AgentFill) -> None:
        self.ledger.record(fill)
        self.agent.on_fill(fill)


def metrics_from(ledger: Ledger, tape: MarketTape, seed: int, attribution_horizon: float) -> RunMetrics:
    """One run's metrics from its ledger and the market tape."""
    path = MidPath.from_tape(tape)
    final_mid = float(path.mids[-1])
    pnl = ledger.attribution(path, attribution_horizon, final_mid)
    return RunMetrics(
        seed=seed,
        pnl=pnl.total,
        spread_capture=pnl.spread_capture,
        adverse_selection=pnl.adverse_selection,
        inventory_pnl=pnl.inventory,
        fees=pnl.fees,
        fills=len(ledger.fills),
        volume=ledger.volume,
        max_abs_inventory=ledger.max_abs_inventory,
        final_inventory=ledger.inventory,
    )


def run_once(
    scenario: Scenario,
    agent: Agent,
    seed: int,
    attribution_horizon: float = 1.0,
    maker_fee: float = 0.0,
    taker_fee: float = 0.0,
) -> RunMetrics:
    ledger = Ledger(maker_fee=maker_fee, taker_fee=taker_fee)
    simulator = MarketSimulator(
        scenario.params,
        scenario.marks,
        scenario.initial_depth,
        scenario.config(seed),
        agents=(_Recorded(agent, ledger),),
    )
    return metrics_from(ledger, simulator.run().tape, seed, attribution_horizon)


def calibrate_avellaneda_stoikov(
    scenario: Scenario,
    gamma: float,
    seeds: Iterable[int],
    sigma_interval: float = 5.0,
    max_ticks: int = 6,
    calibration_horizon: float = 1800.0,
) -> ASParams:
    """AS parameters in ticks, estimated from agent-free runs of the scenario.

    `sigma` comes from realized variance sampled every `sigma_interval`
    seconds, and `kappa` from the decay of the fill curve
    (`agents.estimate_fill_curve`); each is averaged over `seeds`. Sweeps
    past the touch are rare, so the runs last `calibration_horizon`, not the
    episode length. `gamma` is a risk preference, not a market property, so
    it is passed in. Use seeds disjoint from the evaluation seeds.
    """
    sigmas, kappas = [], []
    for seed in seeds:
        config = SimulationConfig(horizon=calibration_horizon, tick=scenario.tick, seed=seed)
        tape = MarketSimulator(scenario.params, scenario.marks, scenario.initial_depth, config).run().tape
        sigmas.append(estimate_sigma(tape, scenario.tick, sigma_interval))
        kappas.append(estimate_fill_curve(tape, scenario.tick, max_ticks)[1])
    if not sigmas:
        raise ValueError("no calibration seeds")
    return ASParams(gamma=gamma, sigma=float(np.mean(sigmas)), kappa=float(np.mean(kappas)))


@dataclass(frozen=True)
class EvaluationTable:
    runs: dict[str, list[RunMetrics]]

    def metric(self, name: str, metric: str) -> list[float]:
        return [float(getattr(run, metric)) for run in self.runs[name]]

    def difference(self, a: str, b: str, metric: str = "pnl", seed: int = 0) -> Difference:
        """Paired difference a - b in `metric` over the shared seeds."""
        if [r.seed for r in self.runs[a]] != [r.seed for r in self.runs[b]]:
            raise ValueError(f"{a!r} and {b!r} were not run on the same seeds")
        return paired_difference(self.metric(a, metric), self.metric(b, metric), seed=seed)

    def summary(self) -> dict[str, dict[str, Any]]:
        """Per agent and metric: mean, standard deviation and 95% CI of the mean."""
        out: dict[str, dict[str, Any]] = {}
        for name in self.runs:
            out[name] = {}
            for metric in METRICS:
                values = np.array(self.metric(name, metric))
                ci = bootstrap_mean(list(values))
                out[name][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "ci95": [ci.low, ci.high],
                }
        return out


def evaluate(
    scenario: Scenario,
    agents: Mapping[str, Callable[[], Agent]],
    seeds: Iterable[int],
    attribution_horizon: float = 1.0,
    maker_fee: float = 0.0,
    taker_fee: float = 0.0,
) -> EvaluationTable:
    """Run every agent (freshly built per run) on every seed."""
    seed_list = list(seeds)
    runs = {
        name: [
            run_once(scenario, make(), seed, attribution_horizon, maker_fee, taker_fee) for seed in seed_list
        ]
        for name, make in agents.items()
    }
    return EvaluationTable(runs)

"""Reproducible experiments: a TOML config in, a self-describing run directory out.

An experiment turns calibration artifacts into a `Scenario` and evaluates
every configured market maker on the held-out *test* seeds. When configured,
it also trains a Double DQN on the *train* seeds and keeps the checkpoint
that does best on the *validation* seeds. Avellaneda-Stoikov parameters are
estimated on separate *calibration* seeds. The four seed ranges must be
disjoint, so no number in the results was tuned on the data that produced
it.

The artifacts are the calibration CLI's outputs: `calibrate-itch` JSON (one
window is selected), its `--marks-out` `.npz`, and a `depth-itch` snapshot.

A run directory holds:

    manifest.json   resolved config, git commit and dirty flag, Python and
                    package versions, AS parameters used, creation time
    results.json    every run's metrics, per-agent summaries with bootstrap
                    CIs, DQN-vs-baseline paired differences, training log
    summary.md      the results table
    policy.pt       the selected DQN weights (when trained)

Every number is a deterministic function of the config and artifacts, so a
rerun reproduces results.json exactly.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import tomllib
from collections.abc import Callable, Iterator, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import MISSING, asdict, dataclass, fields
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from microstructure.agents import AvellanedaStoikovAgent, FixedSpreadAgent
from microstructure.avellaneda_stoikov import ASParams
from microstructure.book import Depth
from microstructure.env import EnvConfig, MarketMakingEnv
from microstructure.evaluation import (
    EvaluationTable,
    RunMetrics,
    Scenario,
    calibrate_avellaneda_stoikov,
    evaluate,
)
from microstructure.flow import FlowMarks, FlowType
from microstructure.hawkes import HawkesParams
from microstructure.simulator import Agent

if TYPE_CHECKING:
    from microstructure.rl import DQNConfig, Policy, TrainingLog


logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """The experiment config is invalid."""


# --- config ----------------------------------------------------------------


@dataclass(frozen=True)
class SeedRange:
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not self.start < self.stop:
            raise ConfigError(f"seed range must have start < stop, got [{self.start}, {self.stop})")

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, self.stop))

    def overlaps(self, other: SeedRange) -> bool:
        return self.start < other.stop and other.start < self.stop


@dataclass(frozen=True)
class Seeds:
    calibration: SeedRange
    train: SeedRange
    validation: SeedRange
    test: SeedRange

    def __post_init__(self) -> None:
        named = [(f.name, getattr(self, f.name)) for f in fields(self)]
        for i, (name_a, a) in enumerate(named):
            for name_b, b in named[i + 1 :]:
                if a.overlaps(b):
                    raise ConfigError(f"seed ranges {name_a} and {name_b} overlap")


@dataclass(frozen=True)
class DataSpec:
    calibration: Path
    window: int
    marks: Path
    depth: Path
    tick: int


@dataclass(frozen=True)
class MarketSpec:
    horizon: float
    latency: float = 0.0
    attribution_horizon: float = 1.0
    maker_fee: float = 0.0
    taker_fee: float = 0.0
    max_distance_ticks: int | None = None  # see FlowMarks.within; None keeps every sample


@dataclass(frozen=True)
class FixedSpreadSpec:
    name: str
    size: int
    half_spread_ticks: int


@dataclass(frozen=True)
class AvellanedaStoikovSpec:
    name: str
    size: int
    gamma: float
    inventory_cap: int | None = None


AgentSpec = FixedSpreadSpec | AvellanedaStoikovSpec
_AGENT_KINDS: dict[str, type[FixedSpreadSpec] | type[AvellanedaStoikovSpec]] = {
    "fixed-spread": FixedSpreadSpec,
    "avellaneda-stoikov": AvellanedaStoikovSpec,
}


@dataclass(frozen=True)
class RLSpec:
    episodes: int
    checkpoint_every: int
    env: dict[str, Any]  # EnvConfig fields other than the scenario
    dqn: DQNConfig


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data: DataSpec
    market: MarketSpec
    seeds: Seeds
    agents: tuple[AgentSpec, ...]
    rl: RLSpec | None
    raw: Mapping[str, Any]  # the parsed TOML, for the manifest


def _take(
    table: Mapping[str, Any], where: str, required: AbstractSet[str], optional: AbstractSet[str] = frozenset()
) -> dict[str, Any]:
    unknown = set(table) - required - optional
    if unknown:
        raise ConfigError(f"unknown key(s) in [{where}]: {sorted(unknown)}")
    missing = required - set(table)
    if missing:
        raise ConfigError(f"missing key(s) in [{where}]: {sorted(missing)}")
    return dict(table)


def _rl_spec(table: Mapping[str, Any]) -> RLSpec:
    try:
        from microstructure.rl import DQNConfig
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ConfigError("[rl] needs PyTorch: pip install -e '.[rl]'") from error
    env_keys = {"step_seconds", "quote_size", "max_inventory", "inventory_penalty", "offsets"}
    dqn_keys = {f.name for f in fields(DQNConfig)}
    values = _take(table, "rl", {"episodes", "checkpoint_every"}, env_keys | dqn_keys)
    env: dict[str, Any] = {k: (tuple(v) if k == "offsets" else v) for k, v in values.items() if k in env_keys}
    dqn: dict[str, Any] = {k: (tuple(v) if k == "hidden" else v) for k, v in values.items() if k in dqn_keys}
    return RLSpec(values["episodes"], values["checkpoint_every"], env, DQNConfig(**dqn))


def load_config(path: Path) -> ExperimentConfig:
    """Parse and validate an experiment TOML file. Paths inside it are
    relative to the file's directory."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    top = _take(raw, "top level", {"name", "data", "market", "seeds", "agents"}, {"rl"})
    data = _take(top["data"], "data", {"calibration", "window", "marks", "depth", "tick"})
    for key in ("calibration", "marks", "depth"):
        data[key] = (base / data[key]).resolve()
    market = _take(
        top["market"],
        "market",
        {"horizon"},
        {"latency", "attribution_horizon", "maker_fee", "taker_fee", "max_distance_ticks"},
    )
    seeds = _take(top["seeds"], "seeds", {"calibration", "train", "validation", "test"})
    agents: list[AgentSpec] = []
    for name, table in top["agents"].items():
        kind = table.get("kind")
        if kind not in _AGENT_KINDS:
            raise ConfigError(
                f"agent {name!r}: unknown kind {kind!r}; expected one of {sorted(_AGENT_KINDS)}"
            )
        spec_type = _AGENT_KINDS[kind]
        spec_fields = {f.name for f in fields(spec_type)} - {"name"}
        required = {f.name for f in fields(spec_type) if f.default is MISSING} - {"name"}
        values = _take(
            {k: v for k, v in table.items() if k != "kind"}, f"agents.{name}", required, spec_fields
        )
        agents.append(spec_type(name=name, **values))
    return ExperimentConfig(
        name=top["name"],
        data=DataSpec(**data),
        market=MarketSpec(**market),
        seeds=Seeds(**{k: SeedRange(*v) for k, v in seeds.items()}),
        agents=tuple(agents),
        rl=_rl_spec(top["rl"]) if "rl" in top else None,
        raw=raw,
    )


# --- scenario ----------------------------------------------------------------


def _params_from_window(window: Mapping[str, Any]) -> HawkesParams:
    alpha = np.asarray(window["alpha"], dtype=np.float64)
    decays = np.asarray(window["decays"], dtype=np.float64)
    beta = np.broadcast_to(decays[:, None, None], alpha.shape)
    return HawkesParams(window["mu"], alpha, beta)


def _marks(path: Path) -> FlowMarks:
    with np.load(path) as arrays:
        sizes = {kind: arrays[f"size_{kind.name}"] for kind in FlowType}
        distances = {kind: arrays[f"distance_{kind.name}"] for kind in (FlowType.LI, FlowType.LD, FlowType.C)}
    return FlowMarks(sizes, distances)


def _depth(path: Path) -> Depth:
    depth = json.loads(path.read_text(encoding="utf-8"))
    bids = [(int(price), int(qty)) for price, qty in depth["bids"]]
    asks = [(int(price), int(qty)) for price, qty in depth["asks"]]
    return bids, asks


def build_scenario(config: ExperimentConfig, horizon: float | None = None) -> Scenario:
    """The simulated market the config describes (`horizon` overrides it)."""
    calibration = json.loads(config.data.calibration.read_text(encoding="utf-8"))
    marks = _marks(config.data.marks)
    if config.market.max_distance_ticks is not None:
        marks = marks.within(config.market.max_distance_ticks)
    return Scenario(
        params=_params_from_window(calibration["windows"][config.data.window]),
        marks=marks,
        initial_depth=_depth(config.data.depth),
        tick=config.data.tick,
        horizon=config.market.horizon if horizon is None else horizon,
        latency=config.market.latency,
    )


# --- running -----------------------------------------------------------------


def _agent_factory(
    spec: AgentSpec, scenario: Scenario, as_params: Mapping[str, ASParams]
) -> Callable[[], Agent]:
    match spec:
        case FixedSpreadSpec():
            return lambda: FixedSpreadAgent(spec.half_spread_ticks, tick=scenario.tick, size=spec.size)
        case AvellanedaStoikovSpec():
            params = as_params[spec.name]
            return lambda: AvellanedaStoikovAgent(
                params,
                tick=scenario.tick,
                size=spec.size,
                horizon=scenario.horizon,
                inventory_cap=spec.inventory_cap,
            )


def _env_config(config: ExperimentConfig, spec: RLSpec, scenario: Scenario) -> EnvConfig:
    return EnvConfig(
        scenario=scenario,
        maker_fee=config.market.maker_fee,
        taker_fee=config.market.taker_fee,
        **spec.env,
    )


def evaluate_policy(
    env: MarketMakingEnv, policy: Policy, seed: int, attribution_horizon: float
) -> RunMetrics:
    """One greedy episode of `policy`, accounted like a baseline run."""
    observation, done = env.reset(seed), False
    while not done:
        observation, _, done, _ = env.step(policy.act(observation))
    return env.episode_metrics(attribution_horizon)


def _train_and_select(
    config: ExperimentConfig, spec: RLSpec, scenario: Scenario
) -> tuple[Policy, TrainingLog, list[dict[str, float]]]:
    from microstructure.rl import train_dqn

    env_config = _env_config(config, spec, scenario)
    validation_env = MarketMakingEnv(env_config)
    checkpoints: list[tuple[float, int, dict[str, Any]]] = []
    history: list[dict[str, float]] = []

    def on_episode(episode: int, _: float, policy: Policy) -> None:
        if (episode + 1) % spec.checkpoint_every:
            return
        pnls = [
            evaluate_policy(validation_env, policy, seed, config.market.attribution_horizon).pnl
            for seed in config.seeds.validation
        ]
        score = float(np.mean(pnls))
        logger.info("episode %d: validation PnL %.0f over %d seeds", episode + 1, score, len(pnls))
        history.append({"episode": episode + 1, "validation_pnl": score})
        state = {k: v.detach().clone() for k, v in policy.network.state_dict().items()}
        checkpoints.append((score, episode + 1, state))

    train_seeds = [
        config.seeds.train.start + k % (config.seeds.train.stop - config.seeds.train.start)
        for k in range(spec.episodes)
    ]
    policy, log = train_dqn(
        lambda: MarketMakingEnv(env_config), spec.dqn, spec.episodes, train_seeds, on_episode
    )
    if checkpoints:
        best = max(checkpoints, key=lambda c: (c[0], -c[1]))  # best validation PnL, earliest on ties
        policy.network.load_state_dict(best[2])
    return policy, log, history


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _manifest(config: ExperimentConfig, as_params: Mapping[str, ASParams]) -> dict[str, Any]:
    packages = {}
    for name in ("microstructure", "numpy", "scipy", "pandas", "torch"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return {
        "config": dict(config.raw),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")) if _git("rev-parse", "HEAD") != "unknown" else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "avellaneda_stoikov": {name: asdict(params) for name, params in as_params.items()},
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _markdown(table: EvaluationTable) -> str:
    summary = table.summary()
    lines = [
        "| agent | PnL mean [95% CI] | spread capture | adverse selection "
        "| inventory PnL | fills | max abs inventory |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, metrics in summary.items():
        pnl = metrics["pnl"]
        lines.append(
            f"| {name} | {pnl['mean']:,.0f} [{pnl['ci95'][0]:,.0f}, {pnl['ci95'][1]:,.0f}] "
            f"| {metrics['spread_capture']['mean']:,.0f} | {metrics['adverse_selection']['mean']:,.0f} "
            f"| {metrics['inventory_pnl']['mean']:,.0f} | {metrics['fills']['mean']:,.1f} "
            f"| {metrics['max_abs_inventory']['mean']:,.0f} |"
        )
    return "\n".join(lines) + "\n\nPnL figures are in price units (1/10000 dollar) x shares.\n"


def _new_run_directory(root: Path, name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(1000):
        candidate = root / (f"{stamp}-{name}" if suffix == 0 else f"{stamp}-{name}-{suffix}")
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("could not create a unique run directory")


def run_experiment(config: ExperimentConfig, out_root: Path) -> Path:
    """Run the experiment and return its run directory."""
    scenario = build_scenario(config)
    as_specs = [spec for spec in config.agents if isinstance(spec, AvellanedaStoikovSpec)]
    by_gamma = {
        gamma: calibrate_avellaneda_stoikov(scenario, gamma, config.seeds.calibration)
        for gamma in sorted({spec.gamma for spec in as_specs})
    }
    as_params = {spec.name: by_gamma[spec.gamma] for spec in as_specs}
    for gamma, params in by_gamma.items():
        logger.info(
            "Avellaneda-Stoikov gamma=%g: sigma=%.3f ticks/sqrt(s), kappa=%.3f /tick",
            gamma,
            params.sigma,
            params.kappa,
        )
    factories = {spec.name: _agent_factory(spec, scenario, as_params) for spec in config.agents}
    market = config.market
    table = evaluate(
        scenario, factories, config.seeds.test, market.attribution_horizon, market.maker_fee, market.taker_fee
    )

    logger.info("baselines evaluated on %d test seeds", len(table.runs[config.agents[0].name]))
    run = _new_run_directory(out_root, config.name)
    results: dict[str, Any] = {}
    if config.rl is not None:
        policy, log, history = _train_and_select(config, config.rl, scenario)
        test_env = MarketMakingEnv(_env_config(config, config.rl, scenario))
        dqn_runs = [
            evaluate_policy(test_env, policy, seed, market.attribution_horizon) for seed in config.seeds.test
        ]
        table = EvaluationTable({**table.runs, "dqn": dqn_runs})
        results["dqn_vs"] = {spec.name: asdict(table.difference("dqn", spec.name)) for spec in config.agents}
        results["training"] = {"episode_returns": log.episode_returns, "validation": history}
        policy.save(run / "policy.pt")

    results = {
        "summary": table.summary(),
        "runs": {name: [asdict(r) for r in runs] for name, runs in table.runs.items()},
        **results,
    }
    (run / "manifest.json").write_text(
        json.dumps(_manifest(config, as_params), indent=2) + "\n", encoding="utf-8"
    )
    (run / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (run / "summary.md").write_text(_markdown(table), encoding="utf-8")
    return run

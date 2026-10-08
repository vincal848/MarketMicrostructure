"""Phase 7: reproducible experiments from TOML configs and calibration artifacts."""

import json
from pathlib import Path

import numpy as np
import pytest
from experiment_fixtures import CONFIG, MID, RL, write_artifacts

from microstructure.evaluation import EvaluationTable, RunMetrics
from microstructure.experiment import (
    ConfigError,
    build_scenario,
    load_config,
    paired_pnl_report,
    run_experiment,
)
from microstructure.flow import FlowType


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    write_artifacts(tmp_path)
    return tmp_path


def test_config_loads_into_validated_dataclasses(workspace: Path) -> None:
    (workspace / "experiment.toml").write_text(CONFIG)
    config = load_config(workspace / "experiment.toml")
    assert config.name == "unit"
    assert [a.name for a in config.agents] == ["fixed", "as"]
    assert list(config.seeds.test) == [300, 301, 302]
    assert config.rl is None


def test_overlapping_seed_ranges_are_rejected(workspace: Path) -> None:
    (workspace / "experiment.toml").write_text(CONFIG.replace("test = [300, 303]", "test = [101, 103]"))
    with pytest.raises(ConfigError, match="overlap"):
        load_config(workspace / "experiment.toml")


def test_unknown_keys_and_agent_kinds_are_rejected(workspace: Path) -> None:
    (workspace / "experiment.toml").write_text(CONFIG.replace("tick = 100", "tick = 100\ntypo = 1"))
    with pytest.raises(ConfigError, match="typo"):
        load_config(workspace / "experiment.toml")
    (workspace / "experiment.toml").write_text(CONFIG.replace('kind = "fixed-spread"', 'kind = "oracle"'))
    with pytest.raises(ConfigError, match="oracle"):
        load_config(workspace / "experiment.toml")


def test_scenario_is_built_from_the_calibration_artifacts(workspace: Path) -> None:
    (workspace / "experiment.toml").write_text(CONFIG)
    scenario = build_scenario(load_config(workspace / "experiment.toml"))
    assert scenario.params.n_types == 6
    assert scenario.params.beta[0, 0, 0] == 2.0
    assert scenario.initial_depth[1][0] == (MID, 300)
    assert list(scenario.marks.distances[FlowType.LD]) == [1, 2, 3]
    assert scenario.horizon == 20.0


def test_experiment_writes_a_reproducible_run_directory(workspace: Path) -> None:
    (workspace / "experiment.toml").write_text(CONFIG)
    run = run_experiment(load_config(workspace / "experiment.toml"), workspace / "runs")
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["config"]["name"] == "unit"
    assert {"git_sha", "python", "packages", "created"} <= set(manifest)
    assert manifest["avellaneda_stoikov"]["as"]["kappa"] > 0
    results = json.loads((run / "results.json").read_text())
    assert set(results["summary"]) == {"fixed", "as"}
    assert all(len(results["runs"][name]) == 3 for name in ("fixed", "as"))
    assert "| fixed |" in (run / "summary.md").read_text()

    again = run_experiment(load_config(workspace / "experiment.toml"), workspace / "runs")
    assert json.loads((again / "results.json").read_text())["runs"] == results["runs"]


def test_experiment_trains_selects_and_tests_a_dqn(workspace: Path) -> None:
    pytest.importorskip("torch")
    (workspace / "experiment.toml").write_text(CONFIG + RL)
    run = run_experiment(load_config(workspace / "experiment.toml"), workspace / "runs")
    results = json.loads((run / "results.json").read_text())
    assert set(results["summary"]) == {"fixed", "as", "dqn"}
    assert len(results["runs"]["dqn"]) == 3
    assert (run / "policy.pt").exists()
    assert set(results["dqn_vs"]) == {"fixed", "as"}
    assert len(results["training"]["episode_returns"]) == 4

    paired = json.loads((run / "manifest.json").read_text())["dqn_paired_pnl"]
    assert set(paired) == {"fixed", "as"}
    for name, entry in paired.items():
        assert entry["seeds"] == [300, 301, 302]
        expected = [
            d["pnl"] - b["pnl"] for d, b in zip(results["runs"]["dqn"], results["runs"][name], strict=True)
        ]
        assert entry["pnl_diff"] == expected
        assert entry["ci95"] == [results["dqn_vs"][name]["low"], results["dqn_vs"][name]["high"]]


def test_a_saved_policy_is_reloaded_without_retraining(workspace: Path) -> None:
    pytest.importorskip("torch")
    (workspace / "experiment.toml").write_text(CONFIG + RL)
    first = run_experiment(load_config(workspace / "experiment.toml"), workspace / "runs")
    saved = first / "policy.pt"
    (workspace / "reload.toml").write_text(CONFIG + RL + f'load_policy = "{saved.as_posix()}"\n')
    second = run_experiment(load_config(workspace / "reload.toml"), workspace / "runs")
    runs = [json.loads((d / "results.json").read_text())["runs"]["dqn"] for d in (first, second)]
    assert runs[0] == runs[1]
    assert json.loads((second / "results.json").read_text())["training"]["episode_returns"] == []


def test_max_distance_ticks_truncates_the_scenario_marks(workspace: Path) -> None:
    capped = CONFIG.replace("horizon = 20.0", "horizon = 20.0\nmax_distance_ticks = 2")
    (workspace / "experiment.toml").write_text(capped)
    scenario = build_scenario(load_config(workspace / "experiment.toml"))
    assert list(scenario.marks.distances[FlowType.LD]) == [1, 2]
    assert list(scenario.marks.distances[FlowType.C]) == [0, 1, 2]


def _table(pnls: dict[str, list[float]]) -> EvaluationTable:
    return EvaluationTable(
        {n: [RunMetrics(i, p, 0, 0, 0, 0, 0, 0, 0, 0) for i, p in enumerate(v)] for n, v in pnls.items()}
    )


def test_paired_report_separates_a_planted_edge_from_noise() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 100.0, 30)
    noise = rng.normal(0.0, 10.0, 30)
    dqn = base + 50 + noise
    tie = base + 50 + rng.normal(0.0, 10.0, 30)  # same expected PnL as the challenger
    report = paired_pnl_report(_table({"dqn": list(dqn), "edge": list(base), "tie": list(tie)}), "dqn")
    assert report["edge"]["beats"]
    assert report["edge"]["ci95"][0] > 0
    assert not report["tie"]["beats"]
    assert report["tie"]["ci95"][0] < 0 < report["tie"]["ci95"][1]
    assert report["edge"]["seeds"] == list(range(30))
    assert np.allclose(report["edge"]["pnl_diff"], 50 + noise)

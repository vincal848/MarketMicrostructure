"""Command-line entry points."""

import json
from pathlib import Path

import itch_writer as w
import numpy as np
from experiment_fixtures import CONFIG, write_artifacts
from mbp_writer import scenario

from microstructure.cli import main

SPY = 7


def _write(path: Path, *messages: bytes) -> Path:
    path.write_bytes(w.frame(w.stock_directory(SPY, "SPY"), w.system_event(1, "Q"), *messages))
    return path


def test_replay_itch_writes_a_clean_report(tmp_path: Path) -> None:
    feed = _write(
        tmp_path / "clean.itch",
        w.add_order(SPY, 10, ref=1, side="B", shares=100, symbol="SPY", price=1_000_000),
        w.add_order(SPY, 11, ref=2, side="S", shares=100, symbol="SPY", price=1_000_100),
        w.order_executed(SPY, 12, ref=1, shares=40),
    )
    out = tmp_path / "report.json"
    assert main(["replay-itch", str(feed), "--symbol", "SPY", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["symbol"] == "SPY"
    assert report["report"]["executions_audited"] == 1
    assert report["report"]["priority_violations"] == 0
    assert report["final_depth"] == {"bids": [[1_000_000, 60]], "asks": [[1_000_100, 100]]}
    assert report["events_per_second"] > 0


def test_replay_itch_exits_nonzero_when_the_audit_finds_problems(tmp_path: Path) -> None:
    feed = _write(tmp_path / "dirty.itch", w.order_delete(SPY, 10, ref=99))
    assert main(["replay-itch", str(feed), "--symbol", "SPY", "--out", str(tmp_path / "r.json")]) == 1


# --- Phase 3: calibration commands --------------------------------------------

TEN_AM_NS = 10 * 3600 * 10**9


def _busy_minute(path: Path) -> Path:
    """One minute (10:00-10:01) of SPY flow: 60 adds, every third partly cancelled."""
    messages = []
    for k in range(60):
        ts = TEN_AM_NS + k * 10**9
        ref = 100 + k
        side = "B" if k % 2 else "S"
        price = 1_000_000 - 100 * (k % 3) if side == "B" else 1_000_100 + 100 * (k % 3)
        messages.append(w.add_order(SPY, ts, ref=ref, side=side, shares=100, symbol="SPY", price=price))
        if k % 3 == 0:
            messages.append(w.order_cancel(SPY, ts + 1000, ref=ref, shares=50))
    return _write(path, *messages)


def test_calibrate_itch_fits_each_window(tmp_path: Path) -> None:
    out, marks_out = tmp_path / "cal.json", tmp_path / "marks.npz"
    args = ["calibrate-itch", str(_busy_minute(tmp_path / "day.itch")), "--symbol", "SPY"]
    args += ["--start", "10:00", "--end", "10:01", "--minutes", "1"]
    args += ["--out", str(out), "--marks-out", str(marks_out)]
    assert main(args) == 0
    result = json.loads(out.read_text())
    assert result["replay"]["unknown_order_refs"] == 0
    assert len(result["windows"]) == 1
    window = result["windows"][0]
    assert sum(window["counts"]) == result["classified_events_in_session"] == 80  # 60 adds + 20 cancels
    assert window["types"] == ["MB", "MS", "LA", "LI", "LD", "C"]

    marks = np.load(marks_out)
    assert int(marks["size_C"].sum()) == 20 * 50


def test_calibrate_mbp10_fits_each_window(tmp_path: Path) -> None:
    csv = tmp_path / "xnas-itch-20251111.mbp-10.csv"
    scenario().to_csv(csv, index=False)
    out = tmp_path / "cal.json"
    args = ["calibrate-mbp10", str(csv), "--start", "09:30", "--end", "09:31", "--minutes", "1"]
    args += ["--out", str(out)]
    assert main(args) == 0
    result = json.loads(out.read_text())
    assert result["classified_events_in_session"] == 10
    assert len(result["windows"]) == 1


# --- Phase 7: snapshot, stylized facts, simulation and experiments -----------


def test_depth_itch_writes_the_book_at_a_time(tmp_path: Path) -> None:
    out = tmp_path / "depth.json"
    feed = _busy_minute(tmp_path / "day.itch")
    assert main(["depth-itch", str(feed), "--symbol", "SPY", "--at", "10:01", "--out", str(out)]) == 0
    depth = json.loads(out.read_text())
    assert depth["bids"]
    assert depth["asks"]
    assert depth["bids"][0][0] < depth["asks"][0][0]


def test_stylized_itch_summarizes_the_real_tape(tmp_path: Path) -> None:
    out = tmp_path / "real.json"
    feed = _busy_minute(tmp_path / "day.itch")
    assert (
        main(
            [
                "stylized-itch",
                str(feed),
                "--symbol",
                "SPY",
                "--start",
                "10:00",
                "--end",
                "10:01",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    summary = json.loads(out.read_text())
    assert summary["quote_updates"] > 0
    assert len(summary["spread_distribution"]) == 10


def test_simulate_checks_rates_and_summarizes_stylized_facts(tmp_path: Path) -> None:
    write_artifacts(tmp_path)
    (tmp_path / "experiment.toml").write_text(CONFIG)
    out = tmp_path / "sim.json"
    assert (
        main(
            [
                "simulate",
                str(tmp_path / "experiment.toml"),
                "--seeds",
                "0:2",
                "--horizon",
                "60",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    result = json.loads(out.read_text())
    assert len(result["rate_ratio"]) == 6  # simulated / stationary intensity, per type
    assert len(result["seeds"]) == 2
    assert "spread_distribution" in result["seeds"][0]["stylized"]


def test_experiment_command_writes_a_run_directory(tmp_path: Path) -> None:
    write_artifacts(tmp_path)
    (tmp_path / "experiment.toml").write_text(CONFIG)
    assert main(["experiment", str(tmp_path / "experiment.toml"), "--out", str(tmp_path / "runs")]) == 0
    (run,) = (tmp_path / "runs").iterdir()
    assert (run / "manifest.json").exists()
    assert (run / "results.json").exists()


def test_bench_reports_throughput(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    assert main(["bench", "--scale", "0.01", "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert {"book_ops_per_second", "replay_events_per_second", "simulator_events_per_second"} <= set(result)
    assert all(v > 0 for v in result.values())

"""Command-line entry points."""

import json
from pathlib import Path

import itch_writer as w

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

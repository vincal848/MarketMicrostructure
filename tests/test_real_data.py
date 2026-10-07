"""Acceptance tests against real market data.

Skipped unless the data file exists locally; never run in CI. Run with

    MICROSTRUCTURE_ITCH=/path/to/01302019.NASDAQ_ITCH50.gz pytest -m data

Results are recorded in docs/RESULTS.md.
"""

import os
from pathlib import Path

import pytest

from microstructure.itch import read_itch
from microstructure.replay import Replayer, replay

pytestmark = pytest.mark.data

ITCH = Path(os.environ.get("MICROSTRUCTURE_ITCH", "data/01302019.NASDAQ_ITCH50.gz"))
SYMBOL = os.environ.get("MICROSTRUCTURE_SYMBOL", "SPY")


@pytest.fixture(scope="module")
def itch_replay() -> Replayer:
    if not ITCH.exists():
        pytest.skip(f"ITCH file not found: {ITCH}")
    return replay(read_itch(ITCH, SYMBOL), audit_continuous_only=True)


def test_m1_full_day_replay_is_clean(itch_replay: Replayer) -> None:
    report = itch_replay.report
    assert report.executions_audited > 0
    assert report.unknown_order_refs == 0, report.first_issues
    assert report.priority_violations == 0, report.first_issues
    assert report.quantity_mismatches == 0, report.first_issues
    assert report.crossing_adds == 0, report.first_issues

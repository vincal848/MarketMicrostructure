"""Phase 3: Databento MBP-10 records -> six-type flow via snapshot diffs.

The fixture is generated here, in Databento's CSV layout, and reproduces the
quirks seen in real XNAS.ITCH MBP-10 files (2025-11-11): an order replace
published only as the add at its new price, adds with side 'N', fills whose
level decrease arrives on a later record (or never as a cancel record), and
trades with side 'N'.
"""

from pathlib import Path

import pandas as pd
from mbp_writer import ET_0930_NS, row, scenario

from microstructure.databento import classify_mbp10, read_mbp10
from microstructure.flow import ClassifiedFlow, FlowType


def _kinds(flow: ClassifiedFlow) -> list[tuple[FlowType, int, int]]:
    return [(FlowType(k), int(q), int(d)) for k, q, d in zip(flow.kind, flow.qty, flow.distance, strict=True)]


def test_snapshot_diffs_reproduce_the_flow() -> None:
    assert _kinds(classify_mbp10(scenario(), tick=100)) == [
        (FlowType.LA, 100, 0),
        (FlowType.LA, 50, 0),
        (FlowType.LD, 30, 2),
        (FlowType.LI, 20, 1),
        (FlowType.C, 30, 2),
        (FlowType.LD, 30, 1),
        (FlowType.MB, 20, 0),
        (FlowType.MB, 15, 0),
        (FlowType.C, 40, 0),
        (FlowType.MB, 7, 0),
    ]


def test_timestamps_are_nanoseconds_since_midnight_new_york() -> None:
    flow = classify_mbp10(scenario(), tick=100)
    assert int(flow.ts[0]) == ET_0930_NS + 1


def test_a_level_sliding_into_the_top_ten_is_not_an_add() -> None:
    full = [(100.00 - 0.01 * i, 10) for i in range(10)]
    after_cancel = [(100.00 - 0.01 * i, 10) for i in range(1, 11)]  # level 11 slides into view
    rows = [
        row(1, "A", "A", 100.05, 10, 1, full, [(100.05, 10)]),
        row(2, "C", "B", 100.00, 10, 2, after_cancel, [(100.05, 10)]),
    ]
    flow = classify_mbp10(pd.DataFrame(rows), tick=100)
    second_row = flow.ts == ET_0930_NS + 2
    assert _kinds(flow)[-1] == (FlowType.C, 10, 0)
    assert int(second_row.sum()) == 1  # the cancel, and no add for the 11th level coming into view


def test_read_mbp10_streams_a_csv_file(tmp_path: Path) -> None:
    path = tmp_path / "xnas-itch-20251111.mbp-10.csv"
    scenario().to_csv(path, index=False)
    flow = read_mbp10(path, tick=100, chunk_rows=4)  # chunking must not split events
    assert _kinds(flow) == _kinds(classify_mbp10(scenario(), tick=100))

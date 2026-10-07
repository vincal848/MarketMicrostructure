"""Phase 3: Databento MBP-10 records -> six-type flow via snapshot diffs.

The fixture is generated here, in Databento's CSV layout, and reproduces the
quirks seen in real XNAS.ITCH MBP-10 files (2025-11-11): an order replace
published only as the add at its new price, adds with side 'N', fills whose
level decrease arrives on a later record (or never as a cancel record), and
trades with side 'N'.
"""

from pathlib import Path

import pandas as pd

from microstructure.databento import classify_mbp10, read_mbp10
from microstructure.flow import ClassifiedFlow, FlowType

UTC_0930_ET = 1_762_871_400_000_000_000  # 2025-11-11 14:30:00 UTC = 09:30 ET (EST)
ET_0930_NS = 9 * 3600 * 10**9 + 30 * 60 * 10**9
LEVELS = 10
Levels = list[tuple[float, int]]


def _row(ts: int, action: str, side: str, price: float, size: int, seq: int, bids: Levels, asks: Levels) -> dict[str, object]:
    row: dict[str, object] = {
        "ts_recv": UTC_0930_ET + ts + 100,
        "ts_event": UTC_0930_ET + ts,
        "rtype": 10,
        "publisher_id": 2,
        "instrument_id": 15144,
        "action": action,
        "side": side,
        "depth": 0,
        "price": price,
        "size": size,
        "flags": 0 if action == "T" else 128,
        "ts_in_delta": 0,
        "sequence": seq,
    }
    for i in range(LEVELS):
        bid = bids[i] if i < len(bids) else (None, 0)
        ask = asks[i] if i < len(asks) else (None, 0)
        row |= {
            f"bid_px_{i:02d}": bid[0],
            f"ask_px_{i:02d}": ask[0],
            f"bid_sz_{i:02d}": bid[1],
            f"ask_sz_{i:02d}": ask[1],
            f"bid_ct_{i:02d}": int(bid[1] > 0),
            f"ask_ct_{i:02d}": int(ask[1] > 0),
        }
    row["symbol"] = "SPY"
    return row


def _scenario() -> pd.DataFrame:
    b1 = [(100.00, 100)]
    a1 = [(100.02, 50)]
    b2 = [(100.00, 100), (99.98, 30)]
    a2 = [(100.01, 20), (100.02, 50)]
    b3 = [(100.00, 100), (99.99, 30)]
    rows = [
        _row(1, "A", "B", 100.00, 100, 1, b1, []),  # LA (empty side)
        _row(2, "A", "A", 100.02, 50, 2, b1, a1),  # LA
        _row(3, "A", "B", 99.98, 30, 3, b2, a1),  # LD, 2 ticks
        _row(4, "A", "N", 100.01, 20, 4, b2, a2),  # side N: LI on the ask, 1 tick
        _row(5, "A", "B", 99.99, 30, 5, b3, a2),  # replace: C at 99.98 (2 ticks) + LD at 99.99 (1 tick)
        _row(6, "T", "B", 100.01, 20, 6, b3, a2),  # MB 20, book not yet updated
        _row(6, "C", "A", 100.01, 20, 6, b3, a1),  # the fill's decrease: not a cancel
        _row(7, "T", "B", 100.02, 10, 7, b3, a1),  # MB ...
        _row(7, "T", "B", 100.02, 5, 8, b3, [(100.02, 40)]),  # ... same aggressor, same time: merged, 15
        _row(7, "C", "A", 100.02, 5, 8, b3, [(100.02, 35)]),  # second fill's decrease
        _row(8, "C", "B", 100.00, 40, 9, [(100.00, 60), (99.99, 30)], [(100.02, 35)]),  # genuine cancel
        _row(9, "T", "N", 100.015, 7, 10, [(100.00, 60), (99.99, 30)], [(100.02, 35)]),  # above mid: MB
    ]
    return pd.DataFrame(rows)


def _kinds(flow: ClassifiedFlow) -> list[tuple[FlowType, int, int]]:
    return [(FlowType(k), int(q), int(d)) for k, q, d in zip(flow.kind, flow.qty, flow.distance, strict=True)]


def test_snapshot_diffs_reproduce_the_flow() -> None:
    assert _kinds(classify_mbp10(_scenario(), tick=100)) == [
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
    flow = classify_mbp10(_scenario(), tick=100)
    assert int(flow.ts[0]) == ET_0930_NS + 1


def test_a_level_sliding_into_the_top_ten_is_not_an_add() -> None:
    full = [(100.00 - 0.01 * i, 10) for i in range(10)]
    after_cancel = [(100.00 - 0.01 * i, 10) for i in range(1, 11)]  # level 11 slides into view
    rows = [
        _row(1, "A", "A", 100.05, 10, 1, full, [(100.05, 10)]),
        _row(2, "C", "B", 100.00, 10, 2, after_cancel, [(100.05, 10)]),
    ]
    flow = classify_mbp10(pd.DataFrame(rows), tick=100)
    second_row = flow.ts == ET_0930_NS + 2
    assert _kinds(flow)[-1] == (FlowType.C, 10, 0)
    assert int(second_row.sum()) == 1  # the cancel, and no add for the 11th level coming into view


def test_read_mbp10_streams_a_csv_file(tmp_path: Path) -> None:
    path = tmp_path / "xnas-itch-20251111.mbp-10.csv"
    _scenario().to_csv(path, index=False)
    flow = read_mbp10(path, tick=100, chunk_rows=4)  # chunking must not split events
    assert _kinds(flow) == _kinds(classify_mbp10(_scenario(), tick=100))

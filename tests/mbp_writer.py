"""Builds Databento MBP-10 CSV rows for tests, reproducing real-file quirks."""

import pandas as pd

UTC_0930_ET = 1_762_871_400_000_000_000  # 2025-11-11 14:30:00 UTC = 09:30 ET (EST)
ET_0930_NS = 9 * 3600 * 10**9 + 30 * 60 * 10**9
LEVELS = 10
Levels = list[tuple[float, int]]


def row(
    ts: int, action: str, side: str, price: float, size: int, seq: int, bids: Levels, asks: Levels
) -> dict[str, object]:
    record: dict[str, object] = {
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
        record |= {
            f"bid_px_{i:02d}": bid[0],
            f"ask_px_{i:02d}": ask[0],
            f"bid_sz_{i:02d}": bid[1],
            f"ask_sz_{i:02d}": ask[1],
            f"bid_ct_{i:02d}": int(bid[1] > 0),
            f"ask_ct_{i:02d}": int(ask[1] > 0),
        }
    record["symbol"] = "SPY"
    return record


def scenario() -> pd.DataFrame:
    b1 = [(100.00, 100)]
    a1 = [(100.02, 50)]
    b2 = [(100.00, 100), (99.98, 30)]
    a2 = [(100.01, 20), (100.02, 50)]
    b3 = [(100.00, 100), (99.99, 30)]
    rows = [
        row(1, "A", "B", 100.00, 100, 1, b1, []),  # LA (empty side)
        row(2, "A", "A", 100.02, 50, 2, b1, a1),  # LA
        row(3, "A", "B", 99.98, 30, 3, b2, a1),  # LD, 2 ticks
        row(4, "A", "N", 100.01, 20, 4, b2, a2),  # side N: LI on the ask, 1 tick
        row(5, "A", "B", 99.99, 30, 5, b3, a2),  # replace: C at 99.98 (2 ticks) + LD at 99.99 (1 tick)
        row(6, "T", "B", 100.01, 20, 6, b3, a2),  # MB 20, book not yet updated
        row(6, "C", "A", 100.01, 20, 6, b3, a1),  # the fill's decrease: not a cancel
        row(7, "T", "B", 100.02, 10, 7, b3, a1),  # MB ...
        row(7, "T", "B", 100.02, 5, 8, b3, [(100.02, 40)]),  # ... same aggressor, same time: merged, 15
        row(7, "C", "A", 100.02, 5, 8, b3, [(100.02, 35)]),  # second fill's decrease
        row(8, "C", "B", 100.00, 40, 9, [(100.00, 60), (99.99, 30)], [(100.02, 35)]),  # genuine cancel
        row(9, "T", "N", 100.015, 7, 10, [(100.00, 60), (99.99, 30)], [(100.02, 35)]),  # above mid: MB
    ]
    return pd.DataFrame(rows)

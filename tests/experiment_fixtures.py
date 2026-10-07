"""Synthetic calibration artifacts and experiment configs for tests."""

import json
from pathlib import Path

import numpy as np

from microstructure.calibration import DecayGrid, calibrate_window
from microstructure.flow import ClassifiedFlow, FlowType
from microstructure.hawkes import HawkesParams, simulate

SECOND = 1_000_000_000
TEN_AM = 10 * 3600 * SECOND
TICK = 100
MID = 1_000_000


def write_artifacts(directory: Path) -> None:
    """Calibration JSON, marks and depth in the formats the CLI writes."""
    alpha = np.full((6, 6), 0.02)
    np.fill_diagonal(alpha, 0.3)
    truth = HawkesParams(mu=[0.6, 0.6, 1.5, 0.3, 1.2, 2.0], alpha=alpha, beta=np.full((6, 6), 2.0))
    stream = simulate(truth, horizon=600.0, seed=0)
    ts = TEN_AM + np.round(stream.times * SECOND).astype(np.int64)
    ones = np.ones(len(stream), dtype=np.int64)
    flow = ClassifiedFlow(ts=ts, kind=stream.types, qty=100 * ones, distance=ones)
    window = calibrate_window(flow, TEN_AM, TEN_AM + 600 * SECOND, DecayGrid((2.0,)))
    (directory / "calibration.json").write_text(json.dumps({"windows": [window.summary()]}))
    arrays = {f"size_{kind.name}": np.array([100, 200]) for kind in FlowType}
    arrays |= {
        "distance_LI": np.array([1]),
        "distance_LD": np.array([1, 2, 3]),
        "distance_C": np.array([0, 1, 2]),
    }
    np.savez(directory / "marks.npz", **arrays)  # type: ignore[arg-type]
    depth = {
        "bids": [[MID - TICK * (i + 1), 300] for i in range(10)],
        "asks": [[MID + TICK * i, 300] for i in range(10)],
    }
    (directory / "depth.json").write_text(json.dumps(depth))


CONFIG = """
name = "unit"

[data]
calibration = "calibration.json"
window = 0
marks = "marks.npz"
depth = "depth.json"
tick = 100

[market]
horizon = 20.0
latency = 0.001
attribution_horizon = 1.0

[seeds]
calibration = [0, 1]
train = [100, 104]
validation = [200, 202]
test = [300, 303]

[agents.fixed]
kind = "fixed-spread"
half_spread_ticks = 1
size = 100

[agents.as]
kind = "avellaneda-stoikov"
gamma = 0.001
size = 100
inventory_cap = 500
"""

RL = """
[rl]
episodes = 4
checkpoint_every = 2
step_seconds = 1.0
quote_size = 100
max_inventory = 500
inventory_penalty = 0.01
hidden = [8]
batch_size = 8
warmup_steps = 8
epsilon_decay_steps = 40
"""

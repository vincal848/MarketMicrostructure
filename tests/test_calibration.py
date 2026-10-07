"""Phase 3: fitting the six-type Hawkes model to classified flow, per window."""

import json

import numpy as np
import pytest

from microstructure.calibration import calibrate_window, session_windows
from microstructure.flow import ClassifiedFlow, FlowType
from microstructure.hawkes import HawkesParams, simulate

SECOND = 1_000_000_000
TEN_AM = 10 * 3600 * SECOND


def _six_type_truth() -> HawkesParams:
    alpha = np.full((6, 6), 0.05)
    np.fill_diagonal(alpha, 0.6)
    alpha[FlowType.C, FlowType.MS] = 0.8  # sells trigger cancels
    return HawkesParams(mu=[0.3, 0.3, 0.8, 0.2, 0.6, 1.0], alpha=alpha, beta=np.full((6, 6), 4.0))


def _flow(params: HawkesParams, seconds: float, seed: int) -> ClassifiedFlow:
    stream = simulate(params, horizon=seconds, seed=seed)
    ts = TEN_AM + np.round(stream.times * SECOND).astype(np.int64)
    ones = np.ones(len(stream), dtype=np.int64)
    return ClassifiedFlow(ts=ts, kind=stream.types, qty=100 * ones, distance=0 * ones)


def test_session_windows_tile_the_session() -> None:
    windows = session_windows(TEN_AM, TEN_AM + 330 * 60 * SECOND, minutes=30)
    assert len(windows) == 11
    assert windows[0] == (TEN_AM, TEN_AM + 30 * 60 * SECOND)
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:], strict=False))


def test_calibration_recovers_a_known_six_type_process() -> None:
    truth = _six_type_truth()
    flow = _flow(truth, seconds=3000.0, seed=4)
    result = calibrate_window(flow, TEN_AM, TEN_AM + 3000 * SECOND, decay_bounds=(0.5, 50.0))
    assert result.n_events == len(flow)
    assert list(result.counts) == list(np.bincount(flow.kind, minlength=6))
    assert result.hawkes.decay_se is not None
    assert abs(result.hawkes.params.beta[0, 0] - 4.0) < 4 * result.hawkes.decay_se
    assert result.branching_ratio == pytest.approx(result.hawkes.params.spectral_radius())
    assert result.aic_improvement > 0
    assert len(result.hawkes_ks) == len(result.poisson_ks) == 6


def test_a_type_with_no_events_is_tolerated() -> None:
    truth = _six_type_truth()
    flow = _flow(truth, seconds=600.0, seed=5)
    keep = flow.kind != FlowType.LI
    sparse = ClassifiedFlow(ts=flow.ts[keep], kind=flow.kind[keep], qty=flow.qty[keep], distance=flow.distance[keep])
    result = calibrate_window(sparse, TEN_AM, TEN_AM + 600 * SECOND, decay_bounds=(0.5, 50.0))
    assert result.counts[FlowType.LI] == 0
    assert result.hawkes.params.mu[FlowType.LI] == 0.0
    assert np.isnan(result.hawkes_ks[FlowType.LI].pvalue)


def test_window_fit_summary_is_json_serializable() -> None:
    flow = _flow(_six_type_truth(), seconds=300.0, seed=6)
    summary = calibrate_window(flow, TEN_AM, TEN_AM + 300 * SECOND, decay_bounds=(0.5, 50.0)).summary()
    decoded = json.loads(json.dumps(summary))
    assert decoded["types"] == ["MB", "MS", "LA", "LI", "LD", "C"]
    assert set(decoded) >= {"start_ns", "end_ns", "counts", "decay", "branching_ratio", "mu", "alpha", "ks"}

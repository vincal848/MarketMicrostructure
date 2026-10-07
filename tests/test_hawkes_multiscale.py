"""Phase 3b: sum-of-exponentials kernels for multi-timescale order flow.

Real flow reacts on timescales from microseconds to seconds (2019-01-30
SPY: 61% of inter-event gaps < 100 us, 98% < 100 ms). A single exponential
cannot span that, and its fitted decay ran to the search bound. Each
kernel is therefore a sum of U exponentials with fixed decays:

    phi_ij(t) = sum_u alpha_uij exp(-beta_uij t)
"""

import numpy as np
import pytest

from microstructure.hawkes import EventStream, HawkesParams, OnlineHawkes, log_likelihood, simulate
from microstructure.hawkes_estimation import fit, ks_exponential, rescaled_residuals

FAST, SLOW = 50.0, 0.5

TWO_SCALE = HawkesParams(
    mu=[0.4, 0.3],
    alpha=[[[20.0, 5.0], [10.0, 15.0]], [[0.15, 0.05], [0.05, 0.1]]],
    beta=[[[FAST, FAST], [FAST, FAST]], [[SLOW, SLOW], [SLOW, SLOW]]],
)


def _naive_log_likelihood(stream: EventStream, params: HawkesParams) -> float:
    t, k = stream.times, stream.types
    total = 0.0
    for idx in range(len(t)):
        lam = params.mu[k[idx]]
        for prev in range(idx):
            lam += np.sum(params.alpha[:, k[idx], k[prev]] * np.exp(-params.beta[:, k[idx], k[prev]] * (t[idx] - t[prev])))
        total += np.log(lam)
    compensator = params.mu.sum() * stream.horizon
    for idx in range(len(t)):
        tail = 1.0 - np.exp(-params.beta[:, :, k[idx]] * (stream.horizon - t[idx]))
        compensator += np.sum(params.alpha[:, :, k[idx]] / params.beta[:, :, k[idx]] * tail)
    return float(total - compensator)


@pytest.fixture(scope="module")
def two_scale_stream() -> EventStream:
    return simulate(TWO_SCALE, horizon=6000.0, seed=21)


def test_params_store_components_and_promote_single_kernels() -> None:
    single = HawkesParams([0.2], [[0.5]], [[1.0]])
    assert single.alpha.shape == single.beta.shape == (1, 1, 1)
    assert single.n_components == 1
    assert TWO_SCALE.n_components == 2
    expected = np.array([[20 / FAST + 0.15 / SLOW, 5 / FAST + 0.05 / SLOW], [10 / FAST + 0.05 / SLOW, 15 / FAST + 0.1 / SLOW]])
    np.testing.assert_allclose(TWO_SCALE.branching_matrix, expected)


def test_component_shapes_must_agree() -> None:
    with pytest.raises(ValueError, match="shape"):
        HawkesParams([0.2, 0.2], np.zeros((2, 2, 2)), np.ones((3, 2, 2)))


def test_multiscale_log_likelihood_matches_the_naive_definition() -> None:
    stream = simulate(TWO_SCALE, horizon=60.0, seed=1)
    assert len(stream) > 40
    assert log_likelihood(stream, TWO_SCALE) == pytest.approx(_naive_log_likelihood(stream, TWO_SCALE), rel=1e-9)


def test_simulation_rate_matches_the_stationary_intensity(two_scale_stream: EventStream) -> None:
    rates = two_scale_stream.counts(2) / two_scale_stream.horizon
    assert rates == pytest.approx(TWO_SCALE.stationary_intensity(), rel=0.1)


def test_fit_on_a_decay_grid_recovers_both_scales(two_scale_stream: EventStream) -> None:
    result = fit(two_scale_stream, n_types=2, decay=[FAST, SLOW])
    assert np.all(np.abs(result.params.mu - TWO_SCALE.mu) < 4 * result.mu_se)
    assert np.all(np.abs(result.params.alpha - TWO_SCALE.alpha) < 4 * result.alpha_se)
    assert result.n_params == 2 + 2 * 4


def test_two_scales_beat_either_single_scale_on_aic(two_scale_stream: EventStream) -> None:
    both = fit(two_scale_stream, n_types=2, decay=[FAST, SLOW])
    assert both.aic < fit(two_scale_stream, n_types=2, decay=FAST).aic
    assert both.aic < fit(two_scale_stream, n_types=2, decay=SLOW).aic


def test_residuals_under_the_true_multiscale_model_pass_ks(two_scale_stream: EventStream) -> None:
    for test in ks_exponential(rescaled_residuals(two_scale_stream, TWO_SCALE)):
        assert test.pvalue > 0.01


def test_online_excitation_adds_every_components_jump() -> None:
    online = OnlineHawkes(TWO_SCALE, np.random.default_rng(0))
    online.advance_to(3.0)
    before = online.intensity()
    online.excite(0)
    np.testing.assert_allclose(online.intensity() - before, TWO_SCALE.alpha[:, :, 0].sum(axis=0))

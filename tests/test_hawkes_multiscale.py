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
    alpha=[[[20.0, 5.0], [10.0, 15.0]], [[0.1, 0.025], [0.025, 0.05]]],
    beta=[[[FAST, FAST], [FAST, FAST]], [[SLOW, SLOW], [SLOW, SLOW]]],
)


def _naive_log_likelihood(stream: EventStream, params: HawkesParams) -> float:
    t, k = stream.times, stream.types
    total = 0.0
    for idx in range(len(t)):
        lam = params.mu[k[idx]]
        for prev in range(idx):
            lam += np.sum(
                params.alpha[:, k[idx], k[prev]]
                * np.exp(-params.beta[:, k[idx], k[prev]] * (t[idx] - t[prev]))
            )
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
    expected = np.array(
        [
            [20 / FAST + 0.1 / SLOW, 5 / FAST + 0.025 / SLOW],
            [10 / FAST + 0.025 / SLOW, 15 / FAST + 0.05 / SLOW],
        ]
    )
    np.testing.assert_allclose(TWO_SCALE.branching_matrix, expected)


def test_component_shapes_must_agree() -> None:
    with pytest.raises(ValueError, match="shape"):
        HawkesParams([0.2, 0.2], np.zeros((2, 2, 2)), np.ones((3, 2, 2)))


def test_multiscale_log_likelihood_matches_the_naive_definition() -> None:
    stream = simulate(TWO_SCALE, horizon=60.0, seed=1)
    assert len(stream) > 40
    assert log_likelihood(stream, TWO_SCALE) == pytest.approx(
        _naive_log_likelihood(stream, TWO_SCALE), rel=1e-9
    )


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


# --- Robust convergence (found on real data: fits with slow decays diverged) ---


def test_adding_a_decay_never_lowers_the_fitted_likelihood(two_scale_stream: EventStream) -> None:
    # Nested models: a third, much slower component can only help. On real
    # SPY flow, the old optimizer returned a *lower* likelihood (and once an
    # AIC of 4.5e17) after a slow decay was added.
    base = fit(two_scale_stream, n_types=2, decay=[FAST, SLOW])
    nested = fit(two_scale_stream, n_types=2, decay=[FAST, SLOW, 0.01])
    assert nested.log_likelihood >= base.log_likelihood - 1e-6 * abs(base.log_likelihood)


def test_fitted_compensator_equals_the_event_count(two_scale_stream: EventStream) -> None:
    # At the maximum likelihood the compensator over [0, T] equals the number
    # of events of each type (Euler's identity for this linear model), so a
    # mismatch means the optimizer stopped early.
    from microstructure.hawkes import compensator_tails

    result = fit(two_scale_stream, n_types=2, decay=[FAST, SLOW, 0.01, 1000.0])
    tails = compensator_tails(two_scale_stream, 2, result.params.beta)
    compensator = result.params.mu * two_scale_stream.horizon + (result.params.alpha * tails).sum(axis=(0, 2))
    np.testing.assert_allclose(compensator, two_scale_stream.counts(2), rtol=1e-4)


@pytest.fixture(scope="module")
def wide_scale_stream() -> tuple[HawkesParams, EventStream]:
    # Like real flow: a high event rate and decays spanning six orders of
    # magnitude, so kernel-sum columns differ in scale by ~1e5 and the
    # likelihood is badly conditioned.
    decays = [1e4, 1.0, 0.01]
    params = HawkesParams(
        mu=[50.0],
        alpha=[[[2000.0]], [[0.2]], [[0.001]]],
        beta=[[[d]] for d in decays],
    )
    return params, simulate(params, horizon=300.0, seed=5)


def test_badly_conditioned_fits_still_converge(wide_scale_stream: tuple[HawkesParams, EventStream]) -> None:
    from microstructure.hawkes import compensator_tails

    truth, stream = wide_scale_stream
    result = fit(stream, n_types=1, decay=[1e4, 1.0, 0.01])
    tails = compensator_tails(stream, 1, result.params.beta)
    compensator = result.params.mu * stream.horizon + (result.params.alpha * tails).sum(axis=(0, 2))
    np.testing.assert_allclose(compensator, stream.counts(1), rtol=1e-4)
    assert result.log_likelihood >= log_likelihood(stream, truth)


def test_six_type_flow_with_replace_ties_converges() -> None:
    # Regression: real SPY flow (six types, tied cancel/add pairs from order
    # replaces, decays 1e5 .. 1/s) left L-BFGS-B out of evaluations with
    # fitted compensators 1-10% short of the event counts. This smaller
    # analogue left one type 0.05% short under the old optimizer.
    from microstructure.hawkes import compensator_tails

    decays = np.array([1e5, 1e4, 1e3, 100.0, 10.0, 1.0])
    alpha = np.zeros((6, 6, 6))
    alpha[3] = 15.0 * np.eye(6)
    alpha[4] = 0.6
    alpha[5] = 0.05 * np.eye(6)
    params = HawkesParams(
        mu=[1, 1, 20, 2, 15, 25], alpha=alpha, beta=np.broadcast_to(decays[:, None, None], alpha.shape)
    )
    base = simulate(params, horizon=300.0, seed=2)
    pick = np.random.default_rng(0).uniform(size=len(base)) < 0.08
    replaces = base.times[pick & (base.types == 5)]  # each cancel instantly followed by an add
    times = np.concatenate([base.times, replaces])
    types = np.concatenate([base.types, np.full(len(replaces), 4)])
    order = np.argsort(times, kind="stable")
    stream = EventStream(times[order], types[order], 300.0)

    result = fit(stream, n_types=6, decay=list(decays))
    tails = compensator_tails(stream, 6, result.params.beta)
    compensator = result.params.mu * stream.horizon + (result.params.alpha * tails).sum(axis=(0, 2))
    np.testing.assert_allclose(compensator, stream.counts(6), rtol=1e-4)
    assert result.kkt_residual < 1e-5

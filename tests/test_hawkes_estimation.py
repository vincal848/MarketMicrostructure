"""Phase 2: Hawkes maximum likelihood, uncertainty and goodness of fit."""

import numpy as np
import pytest

from microstructure.hawkes import EventStream, HawkesParams, log_likelihood, simulate
from microstructure.hawkes_estimation import (
    fit,
    fit_decay,
    fit_poisson,
    ks_exponential,
    rescaled_residuals,
)

TRUTH_2D = HawkesParams(
    mu=[0.5, 0.3],
    alpha=[[0.8, 0.4], [0.6, 0.6]],
    beta=[[2.0, 2.0], [2.0, 2.0]],
)


def _naive_log_likelihood(stream: EventStream, params: HawkesParams) -> float:
    """O(n^2) oracle straight from the definition, independent of the
    recursions in hawkes.py."""
    t, k = stream.times, stream.types
    total = 0.0
    for idx in range(len(t)):
        i = k[idx]
        lam = params.mu[i]
        for prev in range(idx):
            j = k[prev]
            lam += np.sum(params.alpha[:, i, j] * np.exp(-params.beta[:, i, j] * (t[idx] - t[prev])))
        total += np.log(lam)
    compensator = params.mu.sum() * stream.horizon
    for idx in range(len(t)):
        j = k[idx]
        tail = 1.0 - np.exp(-params.beta[:, :, j] * (stream.horizon - t[idx]))
        compensator += np.sum(params.alpha[:, :, j] / params.beta[:, :, j] * tail)
    return float(total - compensator)


@pytest.fixture(scope="module")
def stream_2d() -> EventStream:
    return simulate(TRUTH_2D, horizon=4000.0, seed=11)


def test_log_likelihood_matches_the_naive_definition() -> None:
    params = HawkesParams(mu=[0.4, 0.2], alpha=[[0.5, 0.3], [0.1, 0.4]], beta=[[1.5, 3.0], [2.0, 0.7]])
    stream = simulate(params, horizon=150.0, seed=3)
    assert len(stream) > 50
    assert log_likelihood(stream, params) == pytest.approx(_naive_log_likelihood(stream, params), rel=1e-9)


def test_fit_recovers_1d_parameters_within_four_standard_errors() -> None:
    truth = HawkesParams(mu=[0.4], alpha=[[1.2]], beta=[[2.0]])
    stream = simulate(truth, horizon=20000.0, seed=5)
    result = fit(stream, n_types=1, decay=2.0)
    assert abs(result.params.mu[0] - 0.4) < 4 * result.mu_se[0]
    assert abs(result.params.alpha[0, 0] - 1.2) < 4 * result.alpha_se[0, 0]


def test_fit_recovers_2d_parameters_within_four_standard_errors(stream_2d: EventStream) -> None:
    result = fit(stream_2d, n_types=2, decay=2.0)
    assert np.all(np.abs(result.params.mu - TRUTH_2D.mu) < 4 * result.mu_se)
    assert np.all(np.abs(result.params.alpha - TRUTH_2D.alpha) < 4 * result.alpha_se)


def test_fitted_log_likelihood_is_at_least_the_truths(stream_2d: EventStream) -> None:
    result = fit(stream_2d, n_types=2, decay=2.0)
    assert result.log_likelihood >= log_likelihood(stream_2d, TRUTH_2D) - 1e-6
    assert result.log_likelihood == pytest.approx(log_likelihood(stream_2d, result.params), rel=1e-9)


def test_fit_decay_recovers_the_shared_decay(stream_2d: EventStream) -> None:
    result = fit_decay(stream_2d, n_types=2, bounds=(0.1, 50.0))
    assert result.decay_se is not None
    assert abs(result.params.beta[0, 0, 0] - 2.0) < 4 * result.decay_se
    assert np.all(np.abs(result.params.alpha - TRUTH_2D.alpha) < 4 * result.alpha_se)
    assert result.n_params == 2 + 4 + 1


def test_residuals_under_the_true_model_are_unit_exponential(stream_2d: EventStream) -> None:
    for test in ks_exponential(rescaled_residuals(stream_2d, TRUTH_2D)):
        assert test.pvalue > 0.01


def test_residuals_sum_to_the_compensator() -> None:
    # Per type, residuals are increments of the compensator at that type's
    # events, so they sum to the compensator at the last such event.
    stream = EventStream([1.0, 2.5, 4.0], [0, 0, 0], horizon=5.0)
    poisson = HawkesParams([0.8], [[0.0]], [[1.0]])
    (residuals,) = rescaled_residuals(stream, poisson)
    np.testing.assert_allclose(residuals, [0.8, 1.2, 1.2])


def test_poisson_misses_the_clustering_that_hawkes_captures(stream_2d: EventStream) -> None:
    hawkes_fit = fit(stream_2d, n_types=2, decay=2.0)
    poisson_fit = fit_poisson(stream_2d, n_types=2)
    assert hawkes_fit.aic < poisson_fit.aic
    poisson_ks = ks_exponential(rescaled_residuals(stream_2d, poisson_fit.params))
    assert all(test.pvalue < 0.01 for test in poisson_ks)


def test_fit_to_poisson_data_finds_little_excitation() -> None:
    poisson = HawkesParams([0.5, 0.5], [[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]])
    stream = simulate(poisson, horizon=5000.0, seed=8)
    result = fit(stream, n_types=2, decay=1.0)
    assert result.params.spectral_radius() < 0.1


def test_poisson_fit_is_the_event_rate() -> None:
    stream = EventStream([0.5, 1.5, 4.0, 4.5], [0, 1, 0, 0], horizon=10.0)
    result = fit_poisson(stream, n_types=2)
    np.testing.assert_allclose(result.params.mu, [0.3, 0.1])
    np.testing.assert_allclose(result.mu_se, np.sqrt([3, 1]) / 10.0)


def test_fit_rejects_a_non_positive_decay(stream_2d: EventStream) -> None:
    with pytest.raises(ValueError, match="decay"):
        fit(stream_2d, n_types=2, decay=0.0)


def test_fit_rejects_types_beyond_n_types(stream_2d: EventStream) -> None:
    with pytest.raises(ValueError, match="types"):
        fit(stream_2d, n_types=1, decay=2.0)


def test_objective_gradient_matches_finite_differences() -> None:
    # The fitter's per-target objective (private, but its gradient drives
    # every estimate, so it is checked directly).
    from microstructure.hawkes_estimation import _objective

    rng = np.random.default_rng(0)
    design = np.hstack([np.ones((40, 1)), rng.uniform(0.0, 3.0, size=(40, 3))])
    linear = np.array([25.0, 4.0, 7.0, 2.5])
    theta = np.array([0.6, 0.2, 0.05, 0.3])
    _, gradient = _objective(theta, design, linear)
    step = 1e-6
    numeric = [
        (_objective(theta + step * e, design, linear)[0] - _objective(theta - step * e, design, linear)[0])
        / (2 * step)
        for e in np.eye(4)
    ]
    np.testing.assert_allclose(gradient, numeric, rtol=1e-6)


def _badly_scaled_design(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A design shaped like real SPY flow's: 37 columns whose scales span
    seven orders of magnitude, most of them sparse (fast kernels are nonzero
    only right after an event)."""
    rng = np.random.default_rng(seed)
    n, columns = 20_000, 36
    scales = np.logspace(-4, 3, columns)
    active = rng.uniform(size=(n, columns)) < np.linspace(0.02, 1.0, columns)
    kernel = active * rng.lognormal(0.0, 1.0, size=(n, columns)) * scales
    design = np.hstack([np.ones((n, 1)), kernel])
    linear = np.concatenate([[1800.0], kernel.sum(axis=0) * rng.uniform(0.5, 3.0, columns)])
    return design, linear


def test_target_optimizer_meets_the_optimality_conditions_on_a_badly_scaled_design() -> None:
    # Contract: the per-target optimizer must reach the KKT conditions however
    # the columns are scaled (on real SPY flow they span ~7 orders of magnitude).
    from microstructure.hawkes_estimation import _maximize_target, _objective

    design, linear = _badly_scaled_design(0)
    theta, _, _, kkt = _maximize_target(design, linear)
    assert kkt < 1e-6
    _, gradient = _objective(theta, design, linear)
    n = design.shape[0]
    # In scale-free units (each coefficient as its share of the compensator)
    # the KKT conditions are: gradient 0 where the coefficient is positive,
    # and <= 0 where it is held at zero.
    scaled = gradient * linear / n
    positive = theta * linear / n > 1e-9
    assert np.max(np.abs(scaled[positive])) < 1e-6
    assert np.max(scaled[~positive], initial=-1.0) < 1e-6
    assert linear @ theta == pytest.approx(n, rel=1e-6)

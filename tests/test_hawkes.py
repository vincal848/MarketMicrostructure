"""Multivariate Hawkes process: validation, stationarity, Ogata thinning,
log-likelihood."""

import numpy as np
import pytest

from microstructure.hawkes import EventStream, HawkesParams, log_likelihood, simulate


def test_spectral_radius_of_the_branching_matrix() -> None:
    assert HawkesParams([0.2], [[0.5]], [[1.0]]).spectral_radius() == pytest.approx(0.5)
    assert HawkesParams([0.2], [[1.5]], [[1.0]]).spectral_radius() == pytest.approx(1.5)


def test_simulation_refuses_a_non_stationary_branching_matrix() -> None:
    with pytest.raises(ValueError, match="stationary"):
        simulate(HawkesParams([0.2], [[1.5]], [[1.0]]), horizon=10.0, seed=0)
    with pytest.raises(ValueError, match="stationary"):
        simulate(HawkesParams([0.2], [[1.0]], [[1.0]]), horizon=10.0, seed=0)  # exactly 1


@pytest.mark.parametrize(
    ("mu", "alpha", "beta", "message"),
    [
        ([0.2, 0.2, 0.2], [[0.1, 0.1], [0.1, 0.1]], [[1.0, 1.0], [1.0, 1.0]], "shape"),
        ([0.5], [[-0.3]], [[1.0]], "alpha"),
        ([0.5], [[0.3]], [[0.0]], "beta"),
        ([0.0], [[0.3]], [[1.0]], "mu"),
    ],
)
def test_invalid_parameters_are_rejected_at_construction(
    mu: list[float], alpha: list[list[float]], beta: list[list[float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        HawkesParams(mu, alpha, beta)


def test_params_are_immutable() -> None:
    params = HawkesParams([0.2], [[0.5]], [[1.0]])
    with pytest.raises(ValueError, match="read-only"):
        params.alpha[0, 0] = 5.0


def test_simulated_event_rate_matches_the_stationary_intensity() -> None:
    params = HawkesParams([0.3, 0.2], [[0.4, 0.1], [0.2, 0.3]], [[1.0, 1.0], [1.0, 1.0]])
    stream = simulate(params, horizon=20000.0, seed=1)
    observed_rate = stream.counts(params.n_types) / stream.horizon
    assert observed_rate == pytest.approx(params.stationary_intensity(), rel=0.1)


def test_simulation_is_reproducible_from_its_seed() -> None:
    params = HawkesParams([0.3], [[0.5]], [[1.2]])
    first, second = simulate(params, 100.0, seed=7), simulate(params, 100.0, seed=7)
    np.testing.assert_array_equal(first.times, second.times)


def test_log_likelihood_is_higher_at_the_true_parameters_than_a_perturbation() -> None:
    truth = HawkesParams([0.3], [[0.5]], [[1.2]])
    stream = simulate(truth, horizon=5000.0, seed=2)
    perturbed = HawkesParams([0.45], [[0.65]], [[0.9]])
    assert log_likelihood(stream, truth) > log_likelihood(stream, perturbed)


def test_log_likelihood_of_a_poisson_process_matches_the_closed_form() -> None:
    # With alpha = 0 the process is Poisson(mu): L = n log mu - mu T.
    stream = EventStream([0.5, 1.5, 4.0], [0, 0, 0], horizon=5.0)
    poisson = HawkesParams([0.8], [[0.0]], [[1.0]])
    assert log_likelihood(stream, poisson) == pytest.approx(3 * np.log(0.8) - 0.8 * 5.0)


def test_event_stream_rejects_events_outside_the_horizon() -> None:
    # Regression: the likelihood used to return a finite value for an event
    # after T, because nothing checked times against the horizon.
    with pytest.raises(ValueError, match="horizon"):
        EventStream([5.0], [0], horizon=1.0)


def test_event_stream_rejects_unsorted_times() -> None:
    with pytest.raises(ValueError, match="sorted"):
        EventStream([2.0, 1.0], [0, 0], horizon=3.0)


def test_log_likelihood_rejects_types_the_model_does_not_have() -> None:
    stream = EventStream([1.0], [3], horizon=2.0)
    with pytest.raises(ValueError, match="types"):
        log_likelihood(stream, HawkesParams([0.5], [[0.3]], [[1.0]]))

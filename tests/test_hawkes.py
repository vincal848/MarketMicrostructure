"""Multivariate Hawkes process: stationarity, Ogata thinning, log-likelihood."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import hawkes


def test_stationarity_check_matches_spectral_radius_of_the_branching_matrix():
    mu, beta = [0.2], [[1.0]]
    assert hawkes.spectral_radius([[0.5]], beta) == pytest.approx(0.5)
    assert hawkes.spectral_radius([[1.5]], beta) == pytest.approx(1.5)
    hawkes.simulate(mu, [[0.5]], beta, T=10.0, seed=0)  # < 1: must not raise


def test_simulation_refuses_a_non_stationary_branching_matrix():
    with pytest.raises(ValueError):
        hawkes.simulate([0.2], [[1.5]], [[1.0]], T=10.0, seed=0)
    with pytest.raises(ValueError):
        hawkes.simulate([0.2], [[1.0]], [[1.0]], T=10.0, seed=0)  # exactly 1


def test_simulated_event_rate_matches_the_stationary_intensity():
    mu = [0.3, 0.2]
    alpha = [[0.4, 0.1], [0.2, 0.3]]
    beta = [[1.0, 1.0], [1.0, 1.0]]
    T = 20000.0
    events = hawkes.simulate(mu, alpha, beta, T=T, seed=1)
    types = np.array([k for _, k in events])
    observed_rate = np.array([np.sum(types == i) for i in range(2)]) / T
    expected_rate = hawkes.stationary_intensity(mu, alpha, beta)
    assert observed_rate == pytest.approx(expected_rate, rel=0.1)


def test_log_likelihood_is_higher_at_the_true_parameters_than_a_perturbation():
    mu, alpha, beta = [0.3], [[0.5]], [[1.2]]
    events = hawkes.simulate(mu, alpha, beta, T=5000.0, seed=2)
    times = np.array([t for t, _ in events])
    types = np.array([k for _, k in events])

    true_ll = hawkes.log_likelihood(times, types, mu, alpha, beta, T=5000.0)
    perturbed_ll = hawkes.log_likelihood(
        times, types, [0.45], [[0.65]], [[0.9]], T=5000.0)
    assert true_ll > perturbed_ll


def test_fit_mle_is_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        hawkes.fit_mle([0.0], [0], n_types=1)

"""Avellaneda-Stoikov reservation price and optimal spread closed forms."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import agents


def test_reservation_price_equals_mid_at_zero_inventory():
    r = agents.reservation_price(
        mid=100.0, inventory=0, gamma=0.1, sigma=0.02, T=1.0, t=0.0)
    assert r == pytest.approx(100.0)


def test_reservation_price_shifts_against_inventory():
    base = dict(mid=100.0, gamma=0.1, sigma=0.02, T=1.0, t=0.0)
    long_r = agents.reservation_price(inventory=10, **base)
    short_r = agents.reservation_price(inventory=-10, **base)
    assert long_r < base["mid"] < short_r


def test_optimal_spread_is_positive():
    s = agents.optimal_spread(gamma=0.1, sigma=0.02, T=1.0, t=0.0, kappa=1.5)
    assert s > 0


def test_optimal_spread_increases_with_gamma_sigma_squared_time_to_go():
    # Holding gamma and kappa fixed, the Poisson-arrival term of the AS
    # formula is constant, so varying sigma or T - t isolates the first
    # term, gamma * sigma^2 * (T - t), and the spread must increase with it.
    fixed = dict(gamma=0.1, kappa=1.5)
    less_time = agents.optimal_spread(sigma=0.02, T=1.0, t=0.9, **fixed)
    more_time = agents.optimal_spread(sigma=0.02, T=1.0, t=0.0, **fixed)
    assert more_time > less_time

    lower_vol = agents.optimal_spread(sigma=0.01, T=1.0, t=0.0, **fixed)
    higher_vol = agents.optimal_spread(sigma=0.04, T=1.0, t=0.0, **fixed)
    assert higher_vol > lower_vol


def test_optimal_spread_rejects_a_terminal_time_before_now():
    with pytest.raises(ValueError):
        agents.optimal_spread(gamma=0.1, sigma=0.02, T=0.0, t=1.0, kappa=1.5)


def test_agent_loop_is_not_implemented_yet():
    agent = agents.AvellanedaStoikovAgent(gamma=0.1, sigma=0.02, kappa=1.5)
    with pytest.raises(NotImplementedError):
        agent.step(mid=100.0, t=0.0)

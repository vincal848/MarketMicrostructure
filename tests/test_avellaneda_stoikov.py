"""Avellaneda-Stoikov reservation price, optimal spread and quotes."""

import pytest

from microstructure.avellaneda_stoikov import (
    ASParams,
    optimal_spread,
    quotes,
    reservation_price,
)

PARAMS = ASParams(gamma=0.1, sigma=0.02, kappa=1.5)


def test_reservation_price_equals_mid_at_zero_inventory() -> None:
    assert reservation_price(100.0, inventory=0, time_to_go=1.0, params=PARAMS) == pytest.approx(100.0)


def test_reservation_price_shifts_against_inventory() -> None:
    long_r = reservation_price(100.0, inventory=10, time_to_go=1.0, params=PARAMS)
    short_r = reservation_price(100.0, inventory=-10, time_to_go=1.0, params=PARAMS)
    assert long_r < 100.0 < short_r


def test_optimal_spread_matches_the_paper_formula() -> None:
    expected = 0.1 * 0.02**2 * 1.0 + (2 / 0.1) * 0.06453852113757118  # ln(1 + 0.1/1.5)
    assert optimal_spread(1.0, PARAMS) == pytest.approx(expected)


def test_optimal_spread_increases_with_time_to_go_and_volatility() -> None:
    # The fill-intensity term is constant for fixed gamma and kappa, so
    # varying sigma or time-to-go isolates gamma * sigma^2 * time_to_go.
    assert optimal_spread(1.0, PARAMS) > optimal_spread(0.1, PARAMS)
    high_vol = ASParams(gamma=0.1, sigma=0.04, kappa=1.5)
    assert optimal_spread(1.0, high_vol) > optimal_spread(1.0, PARAMS)


def test_quotes_are_centered_on_the_reservation_price() -> None:
    quote = quotes(100.0, inventory=25, time_to_go=0.5, params=PARAMS)
    center = reservation_price(100.0, inventory=25, time_to_go=0.5, params=PARAMS)
    assert (quote.bid + quote.ask) / 2 == pytest.approx(center)
    assert quote.ask - quote.bid == pytest.approx(optimal_spread(0.5, PARAMS))


def test_quotes_rescale_exactly_with_the_price_unit() -> None:
    # Guards the units contract in the module docstring: quoting in cents
    # (c = 100) with gamma/c, sigma*c, kappa/c must give 100x the dollar
    # quotes. The legacy project mixed units here and never filled.
    c = 100.0
    dollars = quotes(678.50, inventory=40, time_to_go=0.7, params=PARAMS)
    cents_params = ASParams(gamma=PARAMS.gamma / c, sigma=PARAMS.sigma * c, kappa=PARAMS.kappa / c)
    cents = quotes(678.50 * c, inventory=40, time_to_go=0.7, params=cents_params)
    assert cents.bid == pytest.approx(dollars.bid * c)
    assert cents.ask == pytest.approx(dollars.ask * c)


@pytest.mark.parametrize(
    ("gamma", "sigma", "kappa"),
    [(0.0, 0.02, 1.5), (0.1, -0.01, 1.5), (0.1, 0.02, 0.0)],
)
def test_invalid_parameters_are_rejected(gamma: float, sigma: float, kappa: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        ASParams(gamma=gamma, sigma=sigma, kappa=kappa)


def test_negative_time_to_go_is_rejected() -> None:
    with pytest.raises(ValueError, match="time_to_go"):
        optimal_spread(-0.1, PARAMS)

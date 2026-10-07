"""Avellaneda-Stoikov (2008) closed-form market-making quotes.

An agent with exponential utility (risk aversion gamma), facing a mid price
that moves as Brownian motion (volatility sigma) and fills that arrive as a
Poisson process whose intensity decays like exp(-kappa * distance from mid),
quotes symmetrically around an inventory-adjusted reservation price.

Units matter, and getting them wrong silently produces quotes that never
fill (this is the defect recorded in legacy/cmu_ml2_meta_dqn/README.md).
Pick one price unit -- dollars or ticks -- and express everything in it:

    mid, quotes   price units
    sigma         price units per sqrt(time unit)
    kappa         1 / price units (fill intensity falls by e per 1/kappa)
    gamma         1 / (price units x shares) (risk aversion per unit wealth)
    time_to_go    time units (T - t in the paper)

Rescaling the price unit by c (e.g. dollars -> cents, c = 100) with
gamma/c, sigma*c and kappa/c rescales every output by exactly c;
`tests/test_avellaneda_stoikov.py` checks this invariance.

The stateful agent loop (requoting each tick, tracking cash and inventory
from book fills, PnL attribution) is Phase 5 work in docs/ROADMAP.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ASParams:
    """Model constants, validated once on construction."""

    gamma: float
    sigma: float
    kappa: float

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma!r}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma!r}")
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa!r}")


@dataclass(frozen=True, slots=True)
class Quote:
    bid: float
    ask: float


def _require_time_to_go(time_to_go: float) -> None:
    if time_to_go < 0:
        raise ValueError(f"time_to_go must be non-negative, got {time_to_go!r}")


def reservation_price(mid: float, inventory: int, time_to_go: float, params: ASParams) -> float:
    """Indifference price r = mid - inventory * gamma * sigma^2 * time_to_go.

    Long inventory pulls r below mid, so the agent leans toward selling;
    short inventory does the reverse.
    """
    _require_time_to_go(time_to_go)
    return mid - inventory * params.gamma * params.sigma**2 * time_to_go


def optimal_spread(time_to_go: float, params: ASParams) -> float:
    """Total spread around r (paper eq. 10):

        gamma * sigma^2 * time_to_go + (2 / gamma) * ln(1 + gamma / kappa)

    The first term is the inventory-risk premium, largest early in the
    horizon or when volatility is high. The second comes from the fill
    model alone and does not depend on time-to-go.
    """
    _require_time_to_go(time_to_go)
    inventory_risk_premium = params.gamma * params.sigma**2 * time_to_go
    fill_intensity_term = (2.0 / params.gamma) * math.log1p(params.gamma / params.kappa)
    return inventory_risk_premium + fill_intensity_term


def quotes(mid: float, inventory: int, time_to_go: float, params: ASParams) -> Quote:
    """Bid and ask: `optimal_spread` centered on `reservation_price`."""
    center = reservation_price(mid, inventory, time_to_go, params)
    half_spread = optimal_spread(time_to_go, params) / 2.0
    return Quote(bid=center - half_spread, ask=center + half_spread)

"""Avellaneda-Stoikov (2008) market-making formulas: the closed-form
reservation price and optimal spread for an agent with exponential utility
facing Poisson order arrivals. Both are pure functions of the state.

The stateful agent loop -- quoting on every tick, tracking inventory and
cash, handling fills from the book, decomposing PnL into spread capture vs
adverse selection -- is M4 work (see README > Milestones); it needs the
engine's event clock and fill callbacks, neither of which exist yet.

Notation follows the paper:
    mid         mid price
    inventory   signed position (positive = long)
    gamma       risk aversion
    sigma       mid-price volatility (same units as mid, per sqrt(time))
    T, t        terminal time and current time
    kappa       order-arrival decay: how fast fill probability falls off
                with distance from mid (their eq. 29)
"""

import math


def reservation_price(mid, inventory, gamma, sigma, T, t):
    """Indifference price r = mid - inventory * gamma * sigma^2 * (T - t).

    A long inventory (inventory > 0) pulls r below mid, so the agent quotes
    more aggressively to sell and less aggressively to buy; a short
    inventory does the reverse.
    """
    if T < t:
        raise ValueError("T must be >= t, got T=%r t=%r" % (T, t))
    return mid - inventory * gamma * sigma ** 2 * (T - t)


def optimal_spread(gamma, sigma, T, t, kappa):
    """Total bid-ask spread around the reservation price (eq. 10):

    gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)

    The first term is the inventory-risk premium: wider near the start of
    the horizon or when volatility is high. The second comes from the
    Poisson fill-intensity model alone and does not depend on time-to-go.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive, got %r" % (gamma,))
    if kappa <= 0:
        raise ValueError("kappa must be positive, got %r" % (kappa,))
    if T < t:
        raise ValueError("T must be >= t, got T=%r t=%r" % (T, t))
    risk_premium = gamma * sigma ** 2 * (T - t)
    fill_intensity_term = (2.0 / gamma) * math.log(1.0 + gamma / kappa)
    return risk_premium + fill_intensity_term


def quotes(mid, inventory, gamma, sigma, T, t, kappa):
    """Bid and ask: `optimal_spread` centered on `reservation_price`."""
    r = reservation_price(mid, inventory, gamma, sigma, T, t)
    spread = optimal_spread(gamma, sigma, T, t, kappa)
    return r - spread / 2.0, r + spread / 2.0


class AvellanedaStoikovAgent:
    """Stateful quoting loop: requote each tick against the simulated book,
    track inventory and cash as fills arrive, and report a PnL decomposition
    into spread capture vs adverse selection.

    This is M4 work. The closed-form pieces above are ready; what is
    missing is the book's fill-notification interface and an inventory cap
    (the "inventory-limited variant" in the README), neither of which is
    designed yet.
    """

    def __init__(self, gamma, sigma, kappa):
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.inventory = 0
        self.cash = 0.0

    def step(self, mid, t):
        raise NotImplementedError(
            "agent loop is M4 work: needs book fill callbacks and "
            "PnL/inventory tracking, not just the closed-form quotes above"
        )

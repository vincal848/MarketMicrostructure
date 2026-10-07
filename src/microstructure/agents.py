"""Baseline market makers and the estimators that calibrate them.

Every agent works in **ticks**. The legacy project's Avellaneda-Stoikov
baseline mixed dollars and ticks, quoted about 65 ticks away and never
filled (legacy/cmu_ml2_meta_dqn/README.md). Here `sigma` and `kappa` are
estimated from the simulator's own tape, in ticks, by `estimate_sigma` and
`estimate_fill_curve`, and the units contract in `avellaneda_stoikov.py`
applies with price unit = 1 tick.

Agents are post-only: a quote that would cross the book is moved back to one
tick inside the touch instead, so a market maker never pays to take
liquidity by accident.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from microstructure.avellaneda_stoikov import ASParams, quotes
from microstructure.book import Side
from microstructure.simulator import Action, AgentFill, MarketView, Quote
from microstructure.stylized import MarketTape, signature_plot


def _post_only(bid: int, ask: int, best_bid: int, best_ask: int, tick: int) -> tuple[int, int]:
    return min(bid, best_ask - tick), max(ask, best_bid + tick)


@dataclass
class AvellanedaStoikovAgent:
    """Quotes the AS reservation price +- half the optimal spread, in ticks,
    rounded outward to the tick grid.

    With `inventory_cap`, order sizes shrink so that a full fill never takes
    |inventory| past the cap, and the side that would breach it is dropped.
    The guarantee holds when the old quote is replaced before it can fill
    again, i.e. with zero order latency; with latency, an in-flight replace
    can overshoot by one order.
    """

    params: ASParams  # in tick units
    tick: int
    size: int
    horizon: float  # seconds; time-to-go = horizon - now
    inventory_cap: int | None = None
    decision_interval: float = 1.0
    inventory: int = field(default=0, init=False)

    def decide(self, view: MarketView) -> list[Action]:
        if view.best_bid is None or view.best_ask is None:
            return []
        mid_ticks = (view.best_bid + view.best_ask) / 2 / self.tick
        time_to_go = max(self.horizon - view.time, 0.0)
        exact = quotes(mid_ticks, self.inventory, time_to_go, self.params)
        bid, ask = math.floor(exact.bid) * self.tick, math.ceil(exact.ask) * self.tick
        bid, ask = _post_only(bid, ask, view.best_bid, view.best_ask, self.tick)
        bid_qty, ask_qty = self.size, self.size
        if self.inventory_cap is not None:
            bid_qty = min(bid_qty, self.inventory_cap - self.inventory)
            ask_qty = min(ask_qty, self.inventory_cap + self.inventory)
        return [
            Quote(bid=(bid, bid_qty) if bid_qty > 0 else None, ask=(ask, ask_qty) if ask_qty > 0 else None)
        ]

    def on_fill(self, fill: AgentFill) -> None:
        self.inventory += fill.qty if fill.side is Side.BID else -fill.qty


@dataclass
class FixedSpreadAgent:
    """Quotes mid +- a fixed number of ticks with no inventory skew: the
    naive baseline every other strategy must beat."""

    half_spread_ticks: int
    tick: int
    size: int
    decision_interval: float = 1.0
    inventory: int = field(default=0, init=False)

    def decide(self, view: MarketView) -> list[Action]:
        if view.best_bid is None or view.best_ask is None:
            return []
        mid_ticks = (view.best_bid + view.best_ask) / 2 / self.tick
        bid = math.floor(mid_ticks - self.half_spread_ticks) * self.tick
        ask = math.ceil(mid_ticks + self.half_spread_ticks) * self.tick
        bid, ask = _post_only(bid, ask, view.best_bid, view.best_ask, self.tick)
        return [Quote(bid=(bid, self.size), ask=(ask, self.size))]

    def on_fill(self, fill: AgentFill) -> None:
        self.inventory += fill.qty if fill.side is Side.BID else -fill.qty


def estimate_sigma(tape: MarketTape, tick: int, interval: float) -> float:
    """Mid volatility in ticks per sqrt(second), from realized variance
    sampled every `interval` seconds (long enough to wash out bid-ask
    bounce; see the signature plot)."""
    return math.sqrt(float(signature_plot(tape, [interval])[0])) / tick


def estimate_fill_curve(
    tape: MarketTape, tick: int, max_ticks: int, min_orders: int = 10
) -> tuple[float, float]:
    """Fit lambda(delta) = A exp(-kappa delta): the rate of market orders that
    reach at least `delta` ticks beyond the touch.

    A resting quote `delta` ticks behind the touch fills only when a market
    order sweeps that deep, so this is the fill intensity Avellaneda-Stoikov
    assumes. Trades sharing a timestamp are one market order, whose depth is
    its deepest fill, measured against the quotes in force just before it.
    Returns (A per second, kappa per tick), fitted by least squares on log
    lambda over depths reached by at least `min_orders` orders.
    """
    if len(tape.trade_times) == 0:
        raise ValueError("tape has no trades")
    order_times = np.unique(tape.trade_times)
    quote_index = np.clip(
        np.searchsorted(tape.times, tape.trade_times, side="right") - 1, 0, len(tape.times) - 1
    )
    touch = np.where(tape.trade_sign > 0, tape.ask[quote_index], tape.bid[quote_index])
    beyond = np.where(tape.trade_sign > 0, tape.trade_prices - touch, touch - tape.trade_prices) / tick
    group = np.searchsorted(order_times, tape.trade_times)
    depth = np.zeros(len(order_times))
    np.maximum.at(depth, group, np.maximum(beyond, 0.0))
    deltas = np.arange(max_ticks)
    reached = np.array([np.sum(depth >= d) for d in deltas])
    usable = reached >= min_orders
    if usable.sum() < 2:
        raise ValueError("too few depths reached to fit a fill curve")
    slope, intercept = np.polyfit(deltas[usable], np.log(reached[usable] / tape.horizon), 1)
    return float(math.exp(intercept)), float(-slope)

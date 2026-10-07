"""Stylized facts of a market, computed identically for real and simulated data.

A `MarketTape` is the minimal common record: the best bid and ask after each
book change (only while both sides exist), and every trade with its
aggressor sign. `record_tape` builds one from replayed order events; the
simulator records one as it runs. The statistics below are the standard
checks of whether a simulated market looks like a real one (Bouchaud et al.
2018):

- the bid-ask spread distribution, in ticks;
- the trade-sign autocorrelation (real order flow is persistent; a Hawkes
  model with short exponential memory reproduces only its short end);
- the signature plot: realized variance of the mid against the sampling
  interval. It is flat for a pure random walk, and microstructure noise
  (bid-ask bounce) inflates it at short intervals;
- trade inter-arrival times.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from microstructure.book import Side
from microstructure.events import ExecuteOrder, ExecuteOrderWithPrice, HiddenTrade, OrderEvent
from microstructure.hawkes import FloatArray, IntArray
from microstructure.replay import Replayer

NS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class MarketTape:
    """Quotes and trades over [0, horizon] seconds."""

    times: FloatArray
    bid: IntArray
    ask: IntArray
    trade_times: FloatArray
    trade_prices: IntArray
    trade_qty: IntArray
    trade_sign: IntArray  # +1 buyer-initiated, -1 seller-initiated
    horizon: float

    @property
    def mid(self) -> FloatArray:
        mid: FloatArray = (self.bid + self.ask) / 2.0
        return mid


def record_tape(events: Iterable[OrderEvent], start_ns: int, end_ns: int) -> MarketTape:
    """Replay `events` and record the tape for start_ns <= ts < end_ns.

    Visible executions are signed by the resting side (a resting ask hit by
    a buyer is +1). Hidden trades are signed by price against the mid, and
    trades exactly at mid are dropped. Cross executions are not trades.
    """
    replayer = Replayer()
    book = replayer.book
    times: list[float] = []
    bids: list[int] = []
    asks: list[int] = []
    trades: list[tuple[float, int, int, int]] = []
    for event in events:
        if event.ts >= end_ns:
            break  # events are time-ordered: nothing later can be inside
        inside = start_ns <= event.ts
        t = (event.ts - start_ns) / NS_PER_SECOND
        if inside:
            match event:
                case ExecuteOrder() if event.order_id in book:
                    order = book.resting_order(event.order_id)
                    trades.append((t, order.price, event.qty, 1 if order.side is Side.ASK else -1))
                case ExecuteOrderWithPrice() if event.printable and event.order_id in book:
                    order = book.resting_order(event.order_id)
                    trades.append((t, event.price, event.qty, 1 if order.side is Side.ASK else -1))
                case HiddenTrade():
                    bid, ask = book.best_bid(), book.best_ask()
                    if bid is not None and ask is not None and 2 * event.price != bid + ask:
                        trades.append((t, event.price, event.qty, 1 if 2 * event.price > bid + ask else -1))
        replayer.apply(event)
        if inside:
            bid, ask = book.best_bid(), book.best_ask()
            if bid is not None and ask is not None:
                times.append(t)
                bids.append(bid)
                asks.append(ask)
    trade_array = np.array(trades, dtype=np.float64).reshape(-1, 4)
    return MarketTape(
        times=np.array(times, dtype=np.float64),
        bid=np.array(bids, dtype=np.int64),
        ask=np.array(asks, dtype=np.int64),
        trade_times=trade_array[:, 0],
        trade_prices=trade_array[:, 1].astype(np.int64),
        trade_qty=trade_array[:, 2].astype(np.int64),
        trade_sign=trade_array[:, 3].astype(np.int64),
        horizon=(end_ns - start_ns) / NS_PER_SECOND,
    )


def spread_distribution(tape: MarketTape, tick: int, max_ticks: int = 10) -> FloatArray:
    """Fraction of quote updates with a spread of 1, 2, ... ticks; the last
    bucket collects every spread of `max_ticks` or more."""
    ticks = np.clip(np.rint((tape.ask - tape.bid) / tick).astype(np.int64), 1, max_ticks)
    counts = np.bincount(ticks - 1, minlength=max_ticks).astype(np.float64)
    distribution: FloatArray = counts / max(counts.sum(), 1.0)
    return distribution


def trade_sign_autocorrelation(signs: IntArray, max_lag: int) -> FloatArray:
    """Sample autocorrelation of trade signs at lags 1 .. max_lag."""
    centered = signs - signs.mean()
    variance = float(np.mean(centered**2))
    if variance == 0:
        return np.full(max_lag, np.nan)
    return np.array(
        [float(np.mean(centered[:-lag] * centered[lag:])) / variance for lag in range(1, max_lag + 1)]
    )


def signature_plot(tape: MarketTape, intervals: Sequence[float]) -> FloatArray:
    """Realized variance of the mid per second, sampling every `interval`.

    The mid at each grid time is the last quote at or before it (the first
    quote before any update). Units: squared price units per second.
    """
    mid = tape.mid
    variances = []
    for interval in intervals:
        grid = np.arange(0.0, tape.horizon + 1e-9, interval)
        index = np.clip(np.searchsorted(tape.times, grid, side="right") - 1, 0, len(mid) - 1)
        sampled = mid[index]
        variances.append(float(np.sum(np.diff(sampled) ** 2)) / tape.horizon)
    return np.array(variances)


def total_variation(p: FloatArray, q: FloatArray) -> float:
    """Total-variation distance between two discrete distributions."""
    return float(0.5 * np.abs(p - q).sum())


def summarize(
    tape: MarketTape, tick: int, intervals: Sequence[float] = (0.1, 1.0, 10.0, 60.0)
) -> dict[str, Any]:
    """The tape's stylized facts as a JSON-ready dict."""
    durations = np.diff(tape.trade_times)
    return {
        "quote_updates": int(tape.times.shape[0]),
        "trades": int(tape.trade_times.shape[0]),
        "mean_spread_ticks": float(np.mean((tape.ask - tape.bid) / tick)) if len(tape.bid) else float("nan"),
        "spread_distribution": spread_distribution(tape, tick).tolist(),
        "trade_sign_acf": trade_sign_autocorrelation(tape.trade_sign, 10).tolist()
        if len(tape.trade_sign) > 10
        else [],
        "signature_intervals": list(intervals),
        "signature_plot": signature_plot(tape, intervals).tolist() if len(tape.times) else [],
        "trade_duration_mean": float(durations.mean()) if len(durations) else float("nan"),
        "trade_duration_cv": float(durations.std() / durations.mean())
        if len(durations) > 1
        else float("nan"),
    }

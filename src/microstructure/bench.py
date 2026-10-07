"""Throughput benchmarks for the hot paths: book operations, replay, simulation.

Each benchmark builds synthetic, seeded input so the numbers are comparable
across machines and commits. `scale` shrinks or grows the workload (1.0 is
about a second per benchmark on a laptop core). Results go to
docs/RESULTS.md alongside the commit they were measured at.
"""

from __future__ import annotations

import time

import numpy as np

from microstructure.book import OrderBook, Side
from microstructure.events import AddOrder, CancelOrder, ExecuteOrder, OrderEvent
from microstructure.flow import FlowMarks, FlowType
from microstructure.hawkes import HawkesParams
from microstructure.replay import replay
from microstructure.simulator import MarketSimulator, SimulationConfig

TICK = 100
MID = 1_000_000


def _book_ops(n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    book = OrderBook()
    started = time.perf_counter()
    for order_id in range(n):
        roll = rng.uniform()
        side = Side.BID if rng.uniform() < 0.5 else Side.ASK
        if roll < 0.6:
            offset = int(rng.integers(0, 20)) * TICK
            price = MID - TICK - offset if side is Side.BID else MID + offset
            book.add_limit_order(order_id, side, price, int(rng.integers(1, 500)))
        elif roll < 0.9 and order_id > 0:
            victim = int(rng.integers(0, order_id))
            if victim in book:
                book.cancel_order(victim)
        else:
            book.market_order(side, int(rng.integers(1, 300)))
    return n / (time.perf_counter() - started)


def _replay_events(n: int, seed: int) -> list[OrderEvent]:
    rng = np.random.default_rng(seed)
    events: list[OrderEvent] = []
    live: list[tuple[int, Side]] = []
    for order_id in range(n):
        if live and rng.uniform() < 0.4:
            index = int(rng.integers(0, len(live)))
            victim, _ = live.pop(index)
            events.append(CancelOrder(ts=order_id, order_id=victim, qty=1))
            continue
        side = Side.BID if rng.uniform() < 0.5 else Side.ASK
        offset = int(rng.integers(0, 20)) * TICK
        price = MID - TICK - offset if side is Side.BID else MID + offset
        events.append(AddOrder(ts=order_id, order_id=order_id, side=side, price=price, qty=10))
        live.append((order_id, side))
    events.append(ExecuteOrder(ts=n, order_id=live[0][0], qty=1))
    return events


def _replay(n: int, seed: int) -> float:
    events = _replay_events(n, seed)
    started = time.perf_counter()
    replay(events)
    return len(events) / (time.perf_counter() - started)


def _simulator(horizon: float, seed: int) -> float:
    alpha = np.full((6, 6), 0.05)
    np.fill_diagonal(alpha, 0.5)
    params = HawkesParams(mu=[5.0, 5.0, 15.0, 3.0, 12.0, 20.0], alpha=alpha, beta=np.full((6, 6), 5.0))
    marks = FlowMarks(
        sizes={kind: np.array([100, 200, 300]) for kind in FlowType},
        distances={
            FlowType.LI: np.array([1]),
            FlowType.LD: np.array([1, 2, 3]),
            FlowType.C: np.array([0, 1, 2]),
        },
    )
    depth = ([(MID - TICK * (i + 1), 500) for i in range(10)], [(MID + TICK * i, 500) for i in range(10)])
    simulator = MarketSimulator(params, marks, depth, SimulationConfig(horizon=horizon, tick=TICK, seed=seed))
    started = time.perf_counter()
    result = simulator.run()
    return float(result.generated.sum()) / (time.perf_counter() - started)


def run_benchmarks(scale: float = 1.0, seed: int = 0) -> dict[str, float]:
    """Events (or operations) per second for each hot path."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale!r}")
    return {
        "book_ops_per_second": _book_ops(max(int(200_000 * scale), 100), seed),
        "replay_events_per_second": _replay(max(int(200_000 * scale), 100), seed),
        "simulator_events_per_second": _simulator(max(300.0 * scale, 5.0), seed),
    }

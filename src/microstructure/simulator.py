"""Discrete-event limit order book simulator with Hawkes background flow.

One clock drives three sources of activity:

1. **Background flow.** `OnlineHawkes` proposes the next six-type event
   before the next scheduled agent activity. Each event becomes a book
   operation, with sizes and distances drawn from calibrated mark samples
   (`flow.FlowMarks`):

       MB / MS   market order of a sampled size against the opposite side
       LA        limit order at the best price of a random side
       LI        limit order d ticks inside the spread (capped to stay inside);
                 an LA when the spread is one tick
       LD        limit order d >= 1 ticks behind the best
       C         cancel (part of) a *background* order whose level is
                 nearest d ticks from the best of a random side

   Passive events pick the bid or ask side with equal probability; the
   six-type alphabet does not record side.

2. **Agent decisions.** Each agent is asked for actions every
   `decision_interval` seconds, and again immediately after each of its fills.

3. **Agent order arrivals.** Actions reach the book `latency` seconds after
   the decision. A `Quote` replaces the agent's resting orders, but an
   identical (price, size) order is left alone so it keeps its queue
   position. An agent market order excites the background flow exactly as a
   background market order of that type would.

Background cancels only ever select background orders, so agents control
their own orders. Every book change is recorded on a `MarketTape`, and every
applied background event on a `ClassifiedFlow`, so the simulated market can
be checked with the same stylized-fact code as the real one.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from microstructure.book import Depth, Fill, Order, OrderBook, Side
from microstructure.flow import N_FLOW_TYPES, ClassifiedFlow, FlowMarks, FlowRecorder, FlowType
from microstructure.hawkes import HawkesParams, IntArray, OnlineHawkes
from microstructure.stylized import MarketTape

NS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class SimulationConfig:
    horizon: float  # seconds
    tick: int  # price units per tick
    seed: int = 0
    latency: float = 0.0  # seconds from decision to arrival at the book

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.tick <= 0 or self.latency < 0:
            raise ValueError(f"invalid simulation config: {self}")


@dataclass(frozen=True)
class OwnOrder:
    side: Side
    price: int
    qty: int
    queue_ahead: int


@dataclass(frozen=True)
class MarketView:
    """What an agent sees when it decides: the book and its own orders."""

    time: float
    best_bid: int | None
    best_ask: int | None
    depth: Depth
    own_orders: tuple[OwnOrder, ...]


@dataclass(frozen=True)
class Quote:
    """Desired resting orders as (price, qty) per side; None means none."""

    bid: tuple[int, int] | None
    ask: tuple[int, int] | None


@dataclass(frozen=True)
class SendMarketOrder:
    side: Side
    qty: int


Action = Quote | SendMarketOrder


@dataclass(frozen=True)
class AgentFill:
    """One execution of an agent order. `side` is the agent's side (BID: it
    bought); `mid` is the mid just before the trade."""

    time: float
    side: Side
    price: int
    qty: int
    aggressive: bool
    mid: float | None


class Agent(Protocol):
    decision_interval: float

    def decide(self, view: MarketView) -> list[Action]: ...

    def on_fill(self, fill: AgentFill) -> None: ...


@dataclass(frozen=True)
class SimulationResult:
    tape: MarketTape
    flow: ClassifiedFlow
    generated: IntArray  # background events proposed by the Hawkes process, per type
    skipped: IntArray  # of those, events that could not be applied (e.g. an empty side)
    order_log: list[tuple[float, float]]  # (decided, arrived) per agent action
    agent_market_orders: int
    excitations: IntArray  # external (agent) excitations per type


_DECIDE, _ARRIVE = 0, 1


class MarketSimulator:
    def __init__(
        self,
        params: HawkesParams,
        marks: FlowMarks,
        initial_depth: Depth,
        config: SimulationConfig,
        agents: Sequence[Agent] = (),
    ) -> None:
        if params.n_types != N_FLOW_TYPES:
            raise ValueError(f"params must have {N_FLOW_TYPES} types, got {params.n_types}")
        self.config = config
        self.marks = marks
        self.agents = list(agents)
        self.rng = np.random.default_rng(config.seed)
        self.hawkes = OnlineHawkes(params, self.rng)
        self.book = OrderBook()
        self._ids = itertools.count(1)
        self._background: dict[Side, dict[int, list[int]]] = {Side.BID: {}, Side.ASK: {}}
        self._agent_of: dict[int, int] = {}  # order id -> agent index
        self._agent_live: list[dict[Side, int]] = [{} for _ in self.agents]
        self._agent_created = [0] * len(self.agents)
        self._queue: list[tuple[float, int, int, int, Action | None]] = []
        self._sequence = itertools.count()
        self._recorder = FlowRecorder()
        self._quotes: tuple[list[float], list[int], list[int]] = ([], [], [])
        self._trades: list[tuple[float, int, int, int]] = []
        self._generated = np.zeros(N_FLOW_TYPES, dtype=np.int64)
        self._skipped = np.zeros(N_FLOW_TYPES, dtype=np.int64)
        self._excitations = np.zeros(N_FLOW_TYPES, dtype=np.int64)
        self._order_log: list[tuple[float, float]] = []
        self._agent_market_orders = 0
        bids, asks = initial_depth
        for side, levels in ((Side.BID, bids), (Side.ASK, asks)):
            for price, qty in levels:
                self._add_background(side, price, qty)
        self._last_mid = self.mid() or 0.0
        for index in range(len(self.agents)):
            self._schedule(0.0, _DECIDE, index, None)

    # --- public ------------------------------------------------------------

    @property
    def now(self) -> float:
        return self.hawkes.now

    def run(self) -> SimulationResult:
        """Simulate to the horizon and return the result."""
        self.advance(self.config.horizon)
        return self.finish()

    def advance(self, until: float) -> None:
        """Process every event strictly before `until` (capped at the
        horizon) and stop the clock there. Agent activity scheduled exactly
        at `until` waits for the next call, so stepping in increments is
        identical to one `run`."""
        until = min(until, self.config.horizon)
        while True:
            next_agent_time = self._queue[0][0] if self._queue else until
            deadline = min(next_agent_time, until)
            background = self.hawkes.next_event(until=deadline)
            if background is not None:
                self._background_event(*background)
                continue
            if deadline >= until:
                return
            time, _, kind, index, action = heapq.heappop(self._queue)
            if kind == _DECIDE:
                self._decide(time, index)
            elif action is not None:
                self._arrive(time, index, action)

    @property
    def last_mid(self) -> float:
        """The most recent two-sided mid (the initial one before any change)."""
        return self._last_mid

    def mid(self) -> float | None:
        bid, ask = self.book.best_bid(), self.book.best_ask()
        return None if bid is None or ask is None else (bid + ask) / 2.0

    def trade_flow_since(self, since: float) -> tuple[int, int]:
        """(buyer-initiated, seller-initiated) traded volume at or after `since`."""
        bought = sold = 0
        for time, _, qty, sign in reversed(self._trades):
            if time < since:
                break
            if sign > 0:
                bought += qty
            else:
                sold += qty
        return bought, sold

    def agent_orders(self, index: int) -> list[Order]:
        """The agent's orders currently resting on the book."""
        return [self.book.resting_order(oid) for oid in self._agent_live[index].values() if oid in self.book]

    def agent_order_count(self, index: int) -> int:
        """How many distinct orders the agent has placed so far."""
        return self._agent_created[index]

    # --- background flow ---------------------------------------------------

    def _background_event(self, time: float, kind_index: int) -> None:
        kind = FlowType(kind_index)
        self._generated[kind] += 1
        applied = False
        if kind in (FlowType.MB, FlowType.MS):
            applied = self._background_market(time, kind)
        elif kind is FlowType.C:
            applied = self._background_cancel(time)
        else:
            applied = self._background_add(time, kind)
        if applied:
            self._record_quote(time)
        else:
            self._skipped[kind] += 1

    def _sample(self, values: IntArray, minimum: int = 0) -> int:
        return max(int(self.rng.choice(values)), minimum) if len(values) else minimum

    def _random_side(self) -> Side:
        return Side.BID if self.rng.uniform() < 0.5 else Side.ASK

    def _background_market(self, time: float, kind: FlowType) -> bool:
        side = Side.BID if kind is FlowType.MB else Side.ASK
        if (self.book.best_ask() if side is Side.BID else self.book.best_bid()) is None:
            return False
        qty = self._sample(self.marks.sizes[kind], minimum=1)
        mid = self.mid()
        fills, leftover = self.book.market_order(side, qty)
        self._settle_fills(time, fills, aggressor_side=side, aggressor_agent=None, mid=mid)
        self._recorder.market(round(time * NS_PER_SECOND), kind, qty - leftover)
        return True

    def _background_add(self, time: float, kind: FlowType) -> bool:
        tick = self.config.tick
        side = self._random_side()
        bid, ask = self.book.best_bid(), self.book.best_ask()
        same = bid if side is Side.BID else ask
        toward_spread = 1 if side is Side.BID else -1
        if same is None:
            other = ask if side is Side.BID else bid
            reference = round(self._last_mid) if other is None else other
            same = reference - toward_spread * tick
            kind = FlowType.LA
        distance = 0
        if kind is FlowType.LI:
            spread_ticks = 0 if bid is None or ask is None else (ask - bid) // tick
            if spread_ticks >= 2:
                distance = min(self._sample(self.marks.distances[FlowType.LI], minimum=1), spread_ticks - 1)
            else:
                kind = FlowType.LA
        elif kind is FlowType.LD:
            distance = self._sample(self.marks.distances[FlowType.LD], minimum=1)
        sign = toward_spread if kind is FlowType.LI else -toward_spread
        price = same + sign * distance * tick
        qty = self._sample(self.marks.sizes[kind], minimum=1)
        self._add_background(side, price, qty)
        self._recorder.passive(round(time * NS_PER_SECOND), kind, qty, distance)
        return True

    def _background_cancel(self, time: float) -> bool:
        first = self._random_side()
        for side in (first, first.opposite):
            levels = self._background[side]
            self._prune(levels)
            if not levels:
                continue
            best = self.book.best_bid() if side is Side.BID else self.book.best_ask()
            assert best is not None  # background levels exist, so the side is not empty
            target = best - self._sample(self.marks.distances[FlowType.C]) * self.config.tick * (
                1 if side is Side.BID else -1
            )
            price = min(levels, key=lambda p: (abs(p - target), p))
            order_id = int(self.rng.choice(levels[price]))
            resting = self.book.resting_order(order_id).qty
            qty = min(self._sample(self.marks.sizes[FlowType.C], minimum=1), resting)
            self.book.cancel_order(order_id, qty)
            distance = abs(best - price) // self.config.tick
            self._recorder.passive(round(time * NS_PER_SECOND), FlowType.C, qty, distance)
            return True
        return False

    def _add_background(self, side: Side, price: int, qty: int) -> None:
        order_id = next(self._ids)
        self.book.add_limit_order(order_id, side, price, qty)
        if order_id in self.book:
            self._background[side].setdefault(price, []).append(order_id)

    def _prune(self, levels: dict[int, list[int]]) -> None:
        for price in list(levels):
            live = [oid for oid in levels[price] if oid in self.book]
            if live:
                levels[price] = live
            else:
                del levels[price]

    # --- agents ------------------------------------------------------------

    def _schedule(self, time: float, kind: int, index: int, action: Action | None) -> None:
        heapq.heappush(self._queue, (time, next(self._sequence), kind, index, action))

    def _view(self, time: float, index: int) -> MarketView:
        own = tuple(
            OwnOrder(order.side, order.price, order.qty, self.book.queue_ahead(order.order_id))
            for order in self.agent_orders(index)
        )
        return MarketView(time, self.book.best_bid(), self.book.best_ask(), self.book.depth_snapshot(5), own)

    def _decide(self, time: float, index: int) -> None:
        agent = self.agents[index]
        for action in agent.decide(self._view(time, index)):
            self._schedule(time + self.config.latency, _ARRIVE, index, action)
            self._order_log.append((time, time + self.config.latency))
        self._schedule(time + agent.decision_interval, _DECIDE, index, None)

    def _arrive(self, time: float, index: int, action: Action) -> None:
        self.hawkes.advance_to(time)
        if isinstance(action, SendMarketOrder):
            self._agent_market(time, index, action)
        else:
            for side, wanted in ((Side.BID, action.bid), (Side.ASK, action.ask)):
                self._requote(time, index, side, wanted)
        self._record_quote(time)

    def _agent_market(self, time: float, index: int, order: SendMarketOrder) -> None:
        kind = FlowType.MB if order.side is Side.BID else FlowType.MS
        mid = self.mid()
        fills, _ = self.book.market_order(order.side, order.qty)
        self._settle_fills(time, fills, aggressor_side=order.side, aggressor_agent=index, mid=mid)
        self.hawkes.excite(kind)
        self._agent_market_orders += 1
        self._excitations[kind] += 1

    def _requote(self, time: float, index: int, side: Side, wanted: tuple[int, int] | None) -> None:
        live = self._agent_live[index]
        current = live.get(side)
        if current is not None and current in self.book:
            order = self.book.resting_order(current)
            if wanted is not None and (order.price, order.qty) == wanted:
                return
            self.book.cancel_order(current)
        live.pop(side, None)
        if wanted is None:
            return
        price, qty = wanted
        order_id = next(self._ids)
        self._agent_of[order_id] = index
        self._agent_created[index] += 1
        mid = self.mid()
        fills = self.book.add_limit_order(order_id, side, price, qty)
        self._settle_fills(time, fills, aggressor_side=side, aggressor_agent=index, mid=mid)
        if order_id in self.book:
            live[side] = order_id

    def _settle_fills(
        self,
        time: float,
        fills: list[Fill],
        aggressor_side: Side,
        aggressor_agent: int | None,
        mid: float | None,
    ) -> None:
        sign = 1 if aggressor_side is Side.BID else -1
        for fill in fills:
            self._trades.append((time, fill.price, fill.qty, sign))
            if aggressor_agent is not None:
                self.agents[aggressor_agent].on_fill(
                    AgentFill(time, aggressor_side, fill.price, fill.qty, True, mid)
                )
            resting_agent = self._agent_of.get(fill.resting_order_id)
            if resting_agent is not None:
                passive = AgentFill(time, aggressor_side.opposite, fill.price, fill.qty, False, mid)
                self.agents[resting_agent].on_fill(passive)
                self._schedule(time, _DECIDE, resting_agent, None)

    # --- recording -----------------------------------------------------------

    def _record_quote(self, time: float) -> None:
        bid, ask = self.book.best_bid(), self.book.best_ask()
        if bid is not None and ask is not None:
            self._last_mid = (bid + ask) / 2.0
            times, bids, asks = self._quotes
            times.append(time)
            bids.append(bid)
            asks.append(ask)

    def finish(self) -> SimulationResult:
        """The run so far as an immutable result."""
        times, bids, asks = self._quotes
        trades = np.array(self._trades, dtype=np.float64).reshape(-1, 4)
        tape = MarketTape(
            times=np.array(times),
            bid=np.array(bids, dtype=np.int64),
            ask=np.array(asks, dtype=np.int64),
            trade_times=trades[:, 0],
            trade_prices=trades[:, 1].astype(np.int64),
            trade_qty=trades[:, 2].astype(np.int64),
            trade_sign=trades[:, 3].astype(np.int64),
            horizon=self.config.horizon,
        )
        return SimulationResult(
            tape=tape,
            flow=self._recorder.finish(),
            generated=self._generated.copy(),
            skipped=self._skipped.copy(),
            order_log=list(self._order_log),
            agent_market_orders=self._agent_market_orders,
            excitations=self._excitations.copy(),
        )

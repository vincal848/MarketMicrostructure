"""A step-based market-making environment over the simulator (Gymnasium style).

Each step is `step_seconds` of simulated market time. The action picks how
many ticks behind the current best bid and best ask to quote, from a small
grid of offsets, so 4 offsets give 16 actions. Inventory skew comes from
choosing asymmetric offsets. Quotes are `quote_size` shares, reduced so that
a full fill never takes |inventory| past `max_inventory`.

Reward per step, in units of one tick on one quote's size:

    r = (MTM_t - MTM_{t-1}) / (tick * quote_size) - lambda * (inventory / quote_size)^2

i.e. mark-to-market PnL minus a quadratic inventory penalty. Spooner et al.
(2018) dampen inventory in this way so the learned policy cannot earn
reward by carrying a directional position.

The observation has 10 base features, all observable by a real participant:
inventory and time-to-go (normalised), spread, L1 and 5-level depth
imbalance, microprice offset, the last step's mid move and signed traded
volume, and the queue ahead of each own quote. `flow_windows` (seconds)
appends, per window, the signed traded-volume imbalance and the mid move
over that trailing window. The Hawkes intensities that drive the simulator
are deliberately *not* observed.

`episode_metrics` reports an episode with the same accounting and attribution
as `evaluation.run_once`, so a learned policy is compared with the
baselines like for like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from microstructure.accounting import Ledger
from microstructure.book import Side
from microstructure.evaluation import RunMetrics, Scenario, metrics_from
from microstructure.simulator import Action, AgentFill, MarketSimulator, MarketView, Quote

OBSERVATION_SIZE = 10


@dataclass(frozen=True)
class EnvConfig:
    scenario: Scenario
    step_seconds: float = 1.0
    quote_size: int = 100
    max_inventory: int = 500
    inventory_penalty: float = 0.01
    offsets: tuple[int, ...] = (0, 1, 2, 4)
    maker_fee: float = 0.0  # per share, price units (negative = rebate)
    taker_fee: float = 0.0
    flow_windows: tuple[float, ...] = ()  # extra trailing windows for flow imbalance and mid move

    def __post_init__(self) -> None:
        if self.step_seconds <= 0 or self.quote_size <= 0 or self.max_inventory < self.quote_size:
            raise ValueError(f"invalid environment config: {self}")


@dataclass
class _EnvAgent:
    """The simulator-side agent: re-sends the env's chosen prices, sized
    against its inventory at the moment it decides."""

    env: MarketMakingEnv
    decision_interval: float
    ledger: Ledger
    prices: tuple[int | None, int | None] = (None, None)

    def decide(self, view: MarketView) -> list[Action]:
        bid, ask = self.prices
        if bid is None and ask is None:
            return []
        return [self.env.sized_quote(bid, ask)]

    def on_fill(self, fill: AgentFill) -> None:
        self.ledger.record(fill)


@dataclass
class _Episode:
    """Everything that exists only between `reset` and the horizon."""

    seed: int
    simulator: MarketSimulator
    agent: _EnvAgent
    previous_mtm: float = 0.0
    previous_mid: float = 0.0
    mids: list[float] = field(default_factory=list)  # mid after each step, starting with the reset mid
    done: bool = False

    def mid(self) -> float:
        mid = self.simulator.mid()
        return mid if mid is not None else self.simulator.last_mid


class MarketMakingEnv:
    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.observation_size = OBSERVATION_SIZE + 2 * len(config.flow_windows)
        self.n_actions = len(config.offsets) ** 2
        self._episode: _Episode | None = None

    # --- actions -----------------------------------------------------------

    def action_index(self, bid_offset: int, ask_offset: int) -> int:
        offsets = self.config.offsets
        return offsets.index(bid_offset) * len(offsets) + offsets.index(ask_offset)

    def quote_for(self, action: int, best_bid: int, best_ask: int) -> Quote:
        """The quote an action means against the given touch."""
        bid_offset, ask_offset = divmod(action, len(self.config.offsets))
        tick = self.config.scenario.tick
        return self.sized_quote(
            best_bid - self.config.offsets[bid_offset] * tick,
            best_ask + self.config.offsets[ask_offset] * tick,
        )

    def sized_quote(self, bid: int | None, ask: int | None) -> Quote:
        inventory = 0 if self._episode is None else self._episode.agent.ledger.inventory
        bid_qty = min(self.config.quote_size, self.config.max_inventory - inventory)
        ask_qty = min(self.config.quote_size, self.config.max_inventory + inventory)
        return Quote(
            bid=(bid, bid_qty) if bid is not None and bid_qty > 0 else None,
            ask=(ask, ask_qty) if ask is not None and ask_qty > 0 else None,
        )

    # --- episode -----------------------------------------------------------

    def reset(self, seed: int) -> np.ndarray:
        scenario = self.config.scenario
        ledger = Ledger(maker_fee=self.config.maker_fee, taker_fee=self.config.taker_fee)
        agent = _EnvAgent(self, decision_interval=self.config.step_seconds, ledger=ledger)
        simulator = MarketSimulator(
            scenario.params, scenario.marks, scenario.initial_depth, scenario.config(seed), agents=(agent,)
        )
        self._episode = _Episode(seed, simulator, agent)
        self._episode.previous_mid = self._episode.mid()
        self._episode.mids.append(self._episode.previous_mid)
        return self._observation(self._episode, mid_move=0.0, since=0.0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        episode = self._episode
        if episode is None or episode.done:
            raise RuntimeError("episode is over: call reset() first")
        simulator, agent = episode.simulator, episode.agent
        bid, ask = simulator.book.best_bid(), simulator.book.best_ask()
        if bid is not None and ask is not None:
            quote = self.quote_for(action, bid, ask)
            agent.prices = (
                None if quote.bid is None else quote.bid[0],
                None if quote.ask is None else quote.ask[0],
            )
        else:
            agent.prices = (None, None)  # one-sided book: stand aside

        start = simulator.now
        end = min(start + self.config.step_seconds, self.config.scenario.horizon)
        simulator.advance(end)
        episode.done = end >= self.config.scenario.horizon

        mid = episode.mid()
        mtm = agent.ledger.mark_to_market(mid)
        scale = self.config.scenario.tick * self.config.quote_size
        inventory = agent.ledger.inventory
        penalty = self.config.inventory_penalty * (inventory / self.config.quote_size) ** 2
        reward = (mtm - episode.previous_mtm) / scale - penalty
        mid_move = (mid - episode.previous_mid) / self.config.scenario.tick
        episode.previous_mtm, episode.previous_mid = mtm, mid
        episode.mids.append(mid)
        info = {"mtm": mtm, "inventory": inventory, "time": end}
        return self._observation(episode, mid_move=mid_move, since=start), float(reward), episode.done, info

    def episode_metrics(self, attribution_horizon: float = 1.0) -> RunMetrics:
        """The finished episode, accounted exactly like `evaluation.run_once`."""
        episode = self._episode
        if episode is None or not episode.done:
            raise RuntimeError("episode is not finished")
        tape = episode.simulator.finish().tape
        return metrics_from(episode.agent.ledger, tape, episode.seed, attribution_horizon)

    # --- observation -------------------------------------------------------

    def _observation(self, episode: _Episode, mid_move: float, since: float) -> np.ndarray:
        simulator, tick = episode.simulator, self.config.scenario.tick
        features = np.zeros(self.observation_size, dtype=np.float32)
        features[0] = episode.agent.ledger.inventory / self.config.max_inventory
        features[1] = max(self.config.scenario.horizon - simulator.now, 0.0) / self.config.scenario.horizon
        bids, asks = simulator.book.depth_snapshot(5)
        if bids and asks:
            (bid, bid_size), (ask, ask_size) = bids[0], asks[0]
            features[2] = (ask - bid) / tick / 10.0
            features[3] = (bid_size - ask_size) / (bid_size + ask_size)
            bid_depth, ask_depth = sum(q for _, q in bids), sum(q for _, q in asks)
            features[4] = (bid_depth - ask_depth) / (bid_depth + ask_depth)
            microprice = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
            features[5] = (microprice - (bid + ask) / 2.0) / tick
        features[6] = float(np.clip(mid_move, -10.0, 10.0)) / 10.0
        bought, sold = simulator.trade_flow_since(since)
        features[7] = (bought - sold) / (bought + sold + 1.0)
        scale = 10.0 * self.config.quote_size
        own = {order.side: order for order in simulator.agent_orders(0)}
        for slot, side in ((8, Side.BID), (9, Side.ASK)):
            order = own.get(side)
            features[slot] = -1.0 if order is None else simulator.book.queue_ahead(order.order_id) / scale
        for k, window in enumerate(self.config.flow_windows):
            bought, sold = simulator.trade_flow_since(simulator.now - window)
            back = min(round(window / self.config.step_seconds), len(episode.mids) - 1)
            move = (episode.mids[-1] - episode.mids[-1 - back]) / tick
            features[OBSERVATION_SIZE + 2 * k] = (bought - sold) / (bought + sold + 1.0)
            features[OBSERVATION_SIZE + 2 * k + 1] = float(np.clip(move, -20.0, 20.0)) / 20.0
        return features

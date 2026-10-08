"""Phase 6: the market-making environment over the simulator."""

import numpy as np
import pytest

from microstructure.env import EnvConfig, MarketMakingEnv
from microstructure.evaluation import Scenario
from microstructure.flow import FlowMarks, FlowType
from microstructure.hawkes import HawkesParams
from microstructure.simulator import MarketSimulator, SimulationConfig

TICK = 100
MID = 1_000_000


def _scenario(horizon: float = 30.0, latency: float = 0.0) -> Scenario:
    alpha = np.full((6, 6), 0.02)
    np.fill_diagonal(alpha, 0.3)
    params = HawkesParams(mu=[0.8, 0.8, 1.5, 0.3, 1.2, 2.0], alpha=alpha, beta=np.full((6, 6), 2.0))
    marks = FlowMarks(
        sizes={kind: np.array([100, 200]) for kind in FlowType},
        distances={
            FlowType.LI: np.array([1]),
            FlowType.LD: np.array([1, 2, 3]),
            FlowType.C: np.array([0, 1, 2]),
        },
    )
    depth = ([(MID - TICK * (i + 1), 300) for i in range(10)], [(MID + TICK * i, 300) for i in range(10)])
    return Scenario(
        params=params, marks=marks, initial_depth=depth, tick=TICK, horizon=horizon, latency=latency
    )


def _env(max_inventory: int = 500) -> MarketMakingEnv:
    return MarketMakingEnv(
        EnvConfig(
            scenario=_scenario(),
            step_seconds=1.0,
            quote_size=100,
            max_inventory=max_inventory,
            inventory_penalty=0.01,
        )
    )


def test_reset_returns_a_finite_observation_of_the_declared_size() -> None:
    env = _env()
    observation = env.reset(seed=0)
    assert observation.shape == (env.observation_size,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))


def test_episode_ends_exactly_at_the_horizon() -> None:
    env = _env()
    env.reset(seed=0)
    steps, done = 0, False
    while not done:
        _, _, done, _ = env.step(0)
        steps += 1
    assert steps == 30
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_same_seed_and_actions_reproduce_the_trajectory() -> None:
    def rollout() -> list[float]:
        env = _env()
        env.reset(seed=4)
        actions = [i % env.n_actions for i in range(30)]
        return [env.step(a)[1] for a in actions]

    assert rollout() == rollout()


def test_reward_is_mark_to_market_change_minus_the_inventory_penalty() -> None:
    env = _env()
    env.reset(seed=1)
    previous = 0.0
    for step in range(30):
        _, reward, done, info = env.step(step % env.n_actions)
        scale = TICK * 100
        expected = (info["mtm"] - previous) / scale - 0.01 * (info["inventory"] / 100) ** 2
        assert reward == pytest.approx(expected)
        previous = info["mtm"]
        if done:
            break


def test_action_offsets_place_quotes_behind_the_touch() -> None:
    env = _env()
    env.reset(seed=2)
    action = env.action_index(bid_offset=2, ask_offset=0)
    quote = env.quote_for(action, best_bid=MID - TICK, best_ask=MID)
    assert quote.bid == (MID - 3 * TICK, 100)
    assert quote.ask == (MID, 100)


def test_inventory_never_exceeds_the_cap_with_zero_latency() -> None:
    env = _env(max_inventory=200)
    env.reset(seed=3)
    always_join = env.action_index(bid_offset=0, ask_offset=0)
    done = False
    while not done:
        _, _, done, info = env.step(always_join)
        assert abs(info["inventory"]) <= 200


def test_episode_metrics_attribute_the_pnl() -> None:
    env = _env()
    env.reset(seed=5)
    done = False
    while not done:
        _, _, done, _ = env.step(env.action_index(bid_offset=0, ask_offset=0))
    metrics = env.episode_metrics(attribution_horizon=1.0)
    assert metrics.fills > 0
    parts = metrics.spread_capture + metrics.adverse_selection + metrics.inventory_pnl - metrics.fees
    assert parts == pytest.approx(metrics.pnl)


def test_simulator_can_advance_in_steps_and_matches_a_single_run() -> None:
    scenario = _scenario(horizon=20.0)
    config = SimulationConfig(horizon=20.0, tick=TICK, seed=9)
    whole = MarketSimulator(scenario.params, scenario.marks, scenario.initial_depth, config).run()
    stepped = MarketSimulator(scenario.params, scenario.marks, scenario.initial_depth, config)
    for t in range(1, 21):
        stepped.advance(float(t))
    np.testing.assert_array_equal(stepped.finish().tape.bid, whole.tape.bid)


def test_flow_windows_append_trailing_features_consistent_with_the_base_ones() -> None:
    config = EnvConfig(scenario=_scenario(), inventory_penalty=0.01, flow_windows=(1.0, 5.0))
    env = MarketMakingEnv(config)
    assert env.observation_size == 14
    env.reset(seed=1)
    for _ in range(8):
        observation, _, _, _ = env.step(0)
    assert observation.shape == (14,)
    assert np.all(np.isfinite(observation))
    # A window of one step is the base step feature (mid move is scaled by 20 instead of 10).
    assert observation[10] == pytest.approx(observation[7])
    assert observation[11] == pytest.approx(observation[6] / 2.0, abs=1e-6)


def test_deviation_actions_are_offsets_from_the_fixed_spread_quote() -> None:
    from microstructure.agents import FixedSpreadAgent
    from microstructure.simulator import MarketView

    config = EnvConfig(scenario=_scenario(), deviation_actions=True, offsets=(-1, 0, 1))
    env = MarketMakingEnv(config)
    env.reset(seed=0)
    bid, ask = MID - TICK, MID  # the initial touch of the test scenario
    fixed = FixedSpreadAgent(1, TICK, 100).decide(MarketView(0.0, bid, ask, ([], []), ()))
    assert env.quote_for(env.action_index(0, 0), bid, ask) == fixed[0]
    wider = env.quote_for(env.action_index(1, 1), bid, ask)
    assert fixed[0].bid is not None
    assert fixed[0].ask is not None
    assert wider.bid is not None
    assert wider.ask is not None
    assert wider.bid[0] == fixed[0].bid[0] - TICK
    assert wider.ask[0] == fixed[0].ask[0] + TICK


def test_residual_reward_subtracts_exactly_the_shadow_fixed_spread_pnl() -> None:
    def total(residual: bool) -> tuple[float, MarketMakingEnv]:
        env = MarketMakingEnv(
            EnvConfig(scenario=_scenario(), inventory_penalty=0.01, residual_reward=residual)
        )
        env.reset(seed=2)
        done, reward_sum = False, 0.0
        while not done:
            _, reward, done, _ = env.step(5)
            reward_sum += reward
        return reward_sum, env

    plain, _ = total(False)
    residual, env = total(True)
    episode = env._episode
    assert episode is not None
    assert episode.shadow is not None
    shadow_mtm = episode.shadow.previous_mtm
    assert shadow_mtm != 0.0  # the shadow trades
    assert plain - residual == pytest.approx(shadow_mtm / (TICK * 100))

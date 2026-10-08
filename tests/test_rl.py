"""Phase 6: Double DQN (optional `rl` extra; skipped without PyTorch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from microstructure.rl import DQNConfig, ReplayBuffer, train_dqn  # noqa: E402


class BanditEnv:
    """One-step episodes: arm 2 pays 1, every other arm pays 0, plus noise."""

    n_actions = 4
    observation_size = 3

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int) -> np.ndarray:
        self._rng = np.random.default_rng(seed)
        return np.ones(3, dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        reward = (1.0 if action == 2 else 0.0) + float(self._rng.normal(0, 0.1))
        return np.ones(3, dtype=np.float32), reward, True, {}


def test_replay_buffer_overwrites_the_oldest_and_samples_batches() -> None:
    buffer = ReplayBuffer(capacity=3, observation_size=2, seed=0)
    for k in range(5):
        buffer.add(
            np.full(2, k, dtype=np.float32), k % 2, float(k), np.full(2, k + 1, dtype=np.float32), k == 4
        )
    assert len(buffer) == 3
    batch = buffer.sample(2)
    assert batch.observations.shape == (2, 2)
    assert set(batch.rewards.tolist()) <= {2.0, 3.0, 4.0}


def test_dqn_learns_the_best_arm_of_a_bandit() -> None:
    config = DQNConfig(
        hidden=(16,), learning_rate=5e-3, batch_size=32, warmup_steps=64, epsilon_decay_steps=300, seed=0
    )
    policy, log = train_dqn(BanditEnv, config, episodes=600, seeds=range(600))
    assert policy.act(np.ones(3, dtype=np.float32)) == 2
    assert len(log.episode_returns) == 600
    assert np.mean(log.episode_returns[-100:]) > 0.8


def test_training_is_reproducible_from_its_seed() -> None:
    config = DQNConfig(hidden=(8,), batch_size=16, warmup_steps=16, epsilon_decay_steps=50, seed=3)
    _, first = train_dqn(BanditEnv, config, episodes=80, seeds=range(80))
    _, second = train_dqn(BanditEnv, config, episodes=80, seeds=range(80))
    assert first.losses == second.losses
    assert first.episode_returns == second.episode_returns


def test_prior_action_is_the_initial_greedy_choice() -> None:
    config = DQNConfig(hidden=(8,), prior_action=3, seed=1)
    policy, _ = train_dqn(BanditEnv, config, episodes=0, seeds=[])
    for scale in (0.1, 1.0, 5.0):
        assert policy.act(np.full(3, scale, dtype=np.float32)) == 3

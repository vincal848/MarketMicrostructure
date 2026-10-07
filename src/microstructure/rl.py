"""Double DQN for discrete-action environments (optional `rl` extra: PyTorch).

Deliberately small and self-contained: a multilayer-perceptron Q-network,
a uniform replay buffer, a target network synced every
`target_update_steps`, epsilon-greedy exploration with a linear decay, and
the Double DQN target (van Hasselt et al. 2016):

    y = r + gamma * (1 - done) * Q_target(s', argmax_a Q_online(s', a))

which removes the overestimation bias of the plain max. The training loop is
owned here rather than taken from a framework, so every source of
randomness is seeded from `DQNConfig.seed` and a run is bit-for-bit
reproducible on the same machine (tests/test_rl.py).

Environments follow the protocol of `env.MarketMakingEnv`: `reset(seed)`
returns an observation, and `step(action)` returns
(observation, reward, done, info).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

FloatArray32 = npt.NDArray[np.float32]


class Env(Protocol):
    n_actions: int
    observation_size: int

    def reset(self, seed: int) -> FloatArray32: ...

    def step(self, action: int) -> tuple[FloatArray32, float, bool, dict[str, Any]]: ...


@dataclass(frozen=True)
class DQNConfig:
    hidden: tuple[int, ...] = (64, 64)
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_capacity: int = 50_000
    warmup_steps: int = 500
    train_every: int = 1
    target_update_steps: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000
    grad_clip: float = 10.0
    seed: int = 0

    def epsilon(self, step: int) -> float:
        fraction = min(step / max(self.epsilon_decay_steps, 1), 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


@dataclass(frozen=True)
class Batch:
    observations: FloatArray32
    actions: npt.NDArray[np.int64]
    rewards: FloatArray32
    next_observations: FloatArray32
    dones: FloatArray32


class ReplayBuffer:
    """Fixed-capacity ring buffer of transitions, sampled uniformly."""

    def __init__(self, capacity: int, observation_size: int, seed: int) -> None:
        self.capacity = capacity
        self._observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self._next = np.zeros((capacity, observation_size), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._size = 0
        self._cursor = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: FloatArray32,
        action: int,
        reward: float,
        next_observation: FloatArray32,
        done: bool,
    ) -> None:
        i = self._cursor
        self._observations[i], self._actions[i], self._rewards[i] = observation, action, reward
        self._next[i], self._dones[i] = next_observation, float(done)
        self._cursor = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        index = self._rng.integers(0, self._size, size=batch_size)
        return Batch(
            self._observations[index],
            self._actions[index],
            self._rewards[index],
            self._next[index],
            self._dones[index],
        )


def _network(observation_size: int, hidden: tuple[int, ...], n_actions: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = observation_size
    for size in hidden:
        layers += [nn.Linear(width, size), nn.ReLU()]
        width = size
    layers.append(nn.Linear(width, n_actions))
    return nn.Sequential(*layers)


class Policy:
    """Greedy policy of a trained Q-network."""

    def __init__(self, network: nn.Sequential) -> None:
        self.network = network

    def q_values(self, observation: FloatArray32) -> FloatArray32:
        with torch.no_grad():
            values = self.network(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))
        result: FloatArray32 = values.squeeze(0).numpy()
        return result

    def act(self, observation: FloatArray32) -> int:
        return int(np.argmax(self.q_values(observation)))

    def save(self, path: Path) -> None:
        torch.save(self.network.state_dict(), path)

    @classmethod
    def load(cls, path: Path, observation_size: int, hidden: tuple[int, ...], n_actions: int) -> Policy:
        network = _network(observation_size, hidden, n_actions)
        network.load_state_dict(torch.load(path, weights_only=True))
        network.eval()
        return cls(network)


@dataclass
class TrainingLog:
    episode_returns: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)


def train_dqn(
    make_env: Callable[[], Env],
    config: DQNConfig,
    episodes: int,
    seeds: Iterable[int],
    on_episode: Callable[[int, float, Policy], None] | None = None,
) -> tuple[Policy, TrainingLog]:
    """Train a Double DQN for `episodes`, resetting episode k with the k-th seed.

    `on_episode(episode, return, policy)` runs after each episode, e.g. to
    checkpoint the policy against validation seeds.
    """
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    env = make_env()
    online = _network(env.observation_size, config.hidden, env.n_actions)
    target = _network(env.observation_size, config.hidden, env.n_actions)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(config.buffer_capacity, env.observation_size, seed=config.seed)
    policy = Policy(online)
    log = TrainingLog()
    step = 0

    for episode, seed in zip(range(episodes), seeds, strict=False):
        observation = env.reset(seed)
        total, done = 0.0, False
        while not done:
            if rng.uniform() < config.epsilon(step):
                action = int(rng.integers(env.n_actions))
            else:
                action = policy.act(observation)
            next_observation, reward, done, _ = env.step(action)
            buffer.add(observation, action, reward, next_observation, done)
            observation, total, step = next_observation, total + reward, step + 1
            if len(buffer) >= config.warmup_steps and step % config.train_every == 0:
                log.losses.append(
                    _train_step(online, target, optimizer, buffer.sample(config.batch_size), config)
                )
            if step % config.target_update_steps == 0:
                target.load_state_dict(online.state_dict())
        log.episode_returns.append(total)
        if on_episode is not None:
            on_episode(episode, total, policy)
    online.eval()
    return policy, log


def _train_step(
    online: nn.Sequential,
    target: nn.Sequential,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    config: DQNConfig,
) -> float:
    observations = torch.as_tensor(batch.observations)
    actions = torch.as_tensor(batch.actions).unsqueeze(1)
    rewards = torch.as_tensor(batch.rewards)
    next_observations = torch.as_tensor(batch.next_observations)
    dones = torch.as_tensor(batch.dones)
    with torch.no_grad():
        best_next = online(next_observations).argmax(dim=1, keepdim=True)
        next_values = target(next_observations).gather(1, best_next).squeeze(1)
        targets = rewards + config.gamma * (1.0 - dones) * next_values
    predicted = online(observations).gather(1, actions).squeeze(1)
    loss = nn.functional.smooth_l1_loss(predicted, targets)
    optimizer.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    nn.utils.clip_grad_norm_(online.parameters(), config.grad_clip)
    optimizer.step()
    return float(loss.item())

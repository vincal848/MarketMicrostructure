"""Multivariate Hawkes process with exponential kernels.

The intensity of event type i at time t, given the history of all K types:

    lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} alpha_ij * exp(-beta_ij (t - t_k^j))

mu is the (K,) background rate; alpha and beta are (K, K) excitation and decay
matrices: a type-j event raises type-i's intensity by alpha_ij at once,
decaying at rate beta_ij. docs/ARCHITECTURE.md maps the K dimensions onto
order-book event types.

The branching matrix n_ij = alpha_ij / beta_ij is the expected number of
type-i events directly triggered by one type-j event. The process is
stationary iff its spectral radius is < 1 (Hawkes & Oakes 1974); `simulate`
refuses to run otherwise, since the expected event count diverges.

Both the simulator and the likelihood carry the same state: an excitation
matrix R[i, j] holding type j's decayed contribution to type i's intensity,
so lambda(t) = mu + R(t).sum(axis=1). Exponential kernels make R Markov --
decaying it to a new time is one elementwise multiply -- which is what keeps
both algorithms O(n_events * K^2) instead of quadratic in events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _frozen(values: npt.ArrayLike, dtype: type[Any]) -> Any:
    array = np.array(values, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class HawkesParams:
    """Validated (mu, alpha, beta) for a K-type exponential Hawkes process.

    Arrays are copied and made read-only on construction, so a params object
    cannot change after the checks below have passed.
    """

    mu: FloatArray
    alpha: FloatArray
    beta: FloatArray

    def __init__(self, mu: npt.ArrayLike, alpha: npt.ArrayLike, beta: npt.ArrayLike) -> None:
        mu_arr = _frozen(mu, np.float64)
        alpha_arr = _frozen(alpha, np.float64)
        beta_arr = _frozen(beta, np.float64)
        k = mu_arr.shape[0] if mu_arr.ndim == 1 else -1
        if k < 1:
            raise ValueError(f"mu must be a non-empty 1-D array, got shape {mu_arr.shape}")
        for name, matrix in (("alpha", alpha_arr), ("beta", beta_arr)):
            if matrix.shape != (k, k):
                raise ValueError(f"{name} must have shape {(k, k)} to match mu, got {matrix.shape}")
        if np.any(mu_arr < 0) or not np.any(mu_arr > 0):
            raise ValueError("mu must be non-negative with at least one positive rate")
        if np.any(alpha_arr < 0):
            raise ValueError("alpha must be non-negative: negative excitation can make intensity negative")
        if np.any(beta_arr <= 0):
            raise ValueError("beta must be strictly positive")
        object.__setattr__(self, "mu", mu_arr)
        object.__setattr__(self, "alpha", alpha_arr)
        object.__setattr__(self, "beta", beta_arr)

    @property
    def n_types(self) -> int:
        return int(self.mu.shape[0])

    @property
    def branching_matrix(self) -> FloatArray:
        branching: FloatArray = self.alpha / self.beta
        return branching

    def spectral_radius(self) -> float:
        """< 1 is necessary and sufficient for stationarity."""
        return float(np.max(np.abs(np.linalg.eigvals(self.branching_matrix))))

    def stationary_intensity(self) -> FloatArray:
        """Long-run mean intensity per type, (I - n)^-1 mu."""
        self._require_stationary()
        intensity: FloatArray = np.linalg.solve(np.eye(self.n_types) - self.branching_matrix, self.mu)
        return intensity

    def _require_stationary(self) -> None:
        rho = self.spectral_radius()
        if rho >= 1:
            raise ValueError(
                f"branching matrix spectral radius {rho:.4f} >= 1: the process is not stationary"
            )


@dataclass(frozen=True)
class EventStream:
    """One observed or simulated realization on [0, horizon].

    `times` is sorted ascending and `types[k]` is the type of the event at
    `times[k]`. Both arrays are read-only.
    """

    times: FloatArray
    types: IntArray
    horizon: float

    def __init__(self, times: npt.ArrayLike, types: npt.ArrayLike, horizon: float) -> None:
        times_arr = _frozen(times, np.float64)
        types_arr = _frozen(types, np.int64)
        if times_arr.ndim != 1 or times_arr.shape != types_arr.shape:
            raise ValueError("times and types must be 1-D arrays of the same length")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon!r}")
        if np.any(np.diff(times_arr) < 0):
            raise ValueError("times must be sorted ascending")
        if times_arr.size and (times_arr[0] < 0 or times_arr[-1] > horizon):
            raise ValueError(f"event times must lie in [0, horizon={horizon}]")
        if np.any(types_arr < 0):
            raise ValueError("event types must be non-negative indices")
        object.__setattr__(self, "times", times_arr)
        object.__setattr__(self, "types", types_arr)
        object.__setattr__(self, "horizon", float(horizon))

    def __len__(self) -> int:
        return int(self.times.shape[0])

    def counts(self, n_types: int) -> IntArray:
        """Number of events of each type 0 .. n_types-1."""
        counts: IntArray = np.bincount(self.types, minlength=n_types).astype(np.int64)
        return counts


def _decayed(excitation: FloatArray, beta: FloatArray, elapsed: float) -> FloatArray:
    decayed: FloatArray = excitation * np.exp(-beta * elapsed)
    return decayed


def simulate(params: HawkesParams, horizon: float, seed: int | None = None) -> EventStream:
    """Simulate one realization on [0, horizon] by Ogata's (1981) thinning.

    With decaying exponential kernels the total intensity is highest right
    after an event and falls until the next one, so the current total
    intensity is a valid upper bound for a thinning proposal. Draw a candidate
    time at that rate; accept it as a type-k event with probability
    lambda_k(candidate) / bound, otherwise keep the candidate time (where the
    intensity is lower, so the next bound is tighter) and try again.
    """
    params._require_stationary()
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon!r}")

    rng = np.random.default_rng(seed)
    excitation = np.zeros((params.n_types, params.n_types))
    last_event_t = 0.0
    t = 0.0
    times: list[float] = []
    types: list[int] = []

    while True:
        excitation_now = _decayed(excitation, params.beta, t - last_event_t)
        upper_bound = float((params.mu + excitation_now.sum(axis=1)).sum())
        t += rng.exponential(1.0 / upper_bound)
        if t >= horizon:
            break
        excitation_at_t = _decayed(excitation, params.beta, t - last_event_t)
        intensity = params.mu + excitation_at_t.sum(axis=1)
        total = float(intensity.sum())
        if rng.uniform() * upper_bound <= total:
            k = int(rng.choice(params.n_types, p=intensity / total))
            times.append(t)
            types.append(k)
            excitation = excitation_at_t
            excitation[:, k] += params.alpha[:, k]
            last_event_t = t

    return EventStream(times, types, horizon)


def log_likelihood(stream: EventStream, params: HawkesParams) -> float:
    """Exact log-likelihood of `stream` under `params`.

    L = sum_k log lambda_{type_k}(t_k) - sum_i integral_0^horizon lambda_i(s) ds

    The first sum uses the Ozaki (1979) recursion on the excitation matrix;
    the compensator (second term) is closed-form for exponential kernels.
    """
    if len(stream) and int(stream.types.max()) >= params.n_types:
        raise ValueError(f"stream has event types outside 0..{params.n_types - 1}")

    excitation = np.zeros((params.n_types, params.n_types))
    last_event_t = 0.0
    log_intensity_sum = 0.0
    for t_k, k in zip(stream.times, stream.types, strict=True):
        excitation = _decayed(excitation, params.beta, t_k - last_event_t)
        intensity_k = params.mu[k] + excitation[k].sum()
        if intensity_k <= 0:
            return float("-inf")
        log_intensity_sum += np.log(intensity_k)
        excitation[:, k] += params.alpha[:, k]
        last_event_t = t_k

    # integral of alpha_ij exp(-beta_ij (s - t)) over [t, horizon], summed
    # over every type-j event t and every target type i.
    compensator = params.mu.sum() * stream.horizon
    for j in range(params.n_types):
        remaining = stream.horizon - stream.times[stream.types == j]
        decay_mass = 1.0 - np.exp(-np.outer(params.beta[:, j], remaining))
        compensator += np.sum(params.branching_matrix[:, j] * decay_mass.sum(axis=1))

    return float(log_intensity_sum - compensator)

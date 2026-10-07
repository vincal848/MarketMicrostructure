"""Multivariate Hawkes process with sum-of-exponentials kernels.

The intensity of event type i at time t, given the history of all K types:

    lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} sum_u alpha_uij * exp(-beta_uij (t - t_k^j))

mu is the (K,) background rate; alpha and beta are (U, K, K): U exponential
components per kernel. A type-j event raises type-i's intensity by
sum_u alpha_uij at once, and component u decays at rate beta_uij. One
component (U = 1) is the classical exponential Hawkes process. Several
components with log-spaced decays let one kernel cover reactions from
microseconds to seconds, which real order flow needs (docs/RESULTS.md, M3).
docs/ARCHITECTURE.md maps the K dimensions onto order-book event types.

The branching matrix n_ij = sum_u alpha_uij / beta_uij is the expected number
of type-i events directly triggered by one type-j event. The process is
stationary iff its spectral radius is < 1 (Hawkes & Oakes 1974); simulation
refuses to run otherwise, since the expected event count diverges.

Simulation and likelihood carry the same state: an excitation array
R[u, i, j] holding component u of type j's decayed contribution to type i's
intensity, so lambda(t) = mu + R(t).sum(axis=(0, 2)). Exponential components
make R Markov -- decaying it to a new time is one elementwise multiply --
which keeps both algorithms linear in the number of events.
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


def _components(values: npt.ArrayLike, k: int) -> FloatArray:
    """(K, K) or (U, K, K) input as a read-only (U, K, K) array."""
    array = np.array(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[np.newaxis]
    if array.ndim != 3 or array.shape[1:] != (k, k) or array.shape[0] < 1:
        raise ValueError(
            f"alpha and beta must have shape {(k, k)} or (U, {k}, {k}) to match mu, got {array.shape}"
        )
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class HawkesParams:
    """Validated (mu, alpha, beta) for a K-type Hawkes process with U
    exponential components per kernel.

    `alpha` and `beta` may be given as (K, K), which means U = 1, or as
    (U, K, K); they are always stored as (U, K, K). Arrays are copied and made
    read-only on construction, so a params object cannot change after the
    checks below have passed.
    """

    mu: FloatArray
    alpha: FloatArray
    beta: FloatArray

    def __init__(self, mu: npt.ArrayLike, alpha: npt.ArrayLike, beta: npt.ArrayLike) -> None:
        mu_arr = _frozen(mu, np.float64)
        k = mu_arr.shape[0] if mu_arr.ndim == 1 else -1
        if k < 1:
            raise ValueError(f"mu must be a non-empty 1-D array, got shape {mu_arr.shape}")
        alpha_arr, beta_arr = _components(alpha, k), _components(beta, k)
        if alpha_arr.shape != beta_arr.shape:
            raise ValueError(f"alpha and beta shapes differ: {alpha_arr.shape} vs {beta_arr.shape}")
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
    def n_components(self) -> int:
        return int(self.alpha.shape[0])

    @property
    def branching_matrix(self) -> FloatArray:
        branching: FloatArray = (self.alpha / self.beta).sum(axis=0)
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
    """Simulate one realization on [0, horizon] by Ogata's (1981) thinning
    (see `OnlineHawkes`)."""
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon!r}")
    online = OnlineHawkes(params, np.random.default_rng(seed))
    times: list[float] = []
    types: list[int] = []
    while (event := online.next_event(until=horizon)) is not None:
        times.append(event[0])
        types.append(event[1])
    return EventStream(times, types, horizon)


# --- Vectorized kernel sums ------------------------------------------------------
#
# The likelihood, the estimators and the goodness-of-fit residuals all need,
# at each event k of a target type i, the kernel sum over earlier events of
# each source type j, for each component u:
#
#     G[k, u, j] = sum_{m < k, type_m = j} exp(-beta_uij (t_k - t_m))
#
# It obeys G_k = exp(-beta dt) (G_{k-1} + [type_{k-1} = j]), evaluated here
# with cumulative sums instead of a Python loop: inside a block starting at
# t0, G_k = exp(-beta (t_k - t0)) * (carry + exclusive cumsum of
# exp(beta (t_m - t0))). Blocks are cut so beta (t - t0) never exceeds
# _MAX_EXPONENT, so the growing factor cannot overflow. "Earlier" means
# earlier in stream order, so events sharing a timestamp excite later ones
# exactly as in the sequential recursion.

_MAX_EXPONENT = 300.0


def _kernel_sum(times: FloatArray, is_source: FloatArray, beta: float) -> FloatArray:
    sums: FloatArray = np.empty_like(times)
    carry = 0.0
    start, n = 0, times.shape[0]
    while start < n:
        t0 = times[start]
        stop = max(int(np.searchsorted(times, t0 + _MAX_EXPONENT / beta, side="right")), start + 1)
        offset = times[start:stop] - t0
        weights = is_source[start:stop] * np.exp(beta * offset)
        exclusive = np.cumsum(weights) - weights
        sums[start:stop] = np.exp(-beta * offset) * (carry + exclusive)
        if stop < n:
            carry = float(np.exp(-beta * (times[stop] - t0)) * (carry + weights.sum()))
        start = stop
    return sums


@dataclass(frozen=True)
class TargetSums:
    """Kernel sums and prior event counts at every event of one target type."""

    times: FloatArray  # (n_i,) times of the target type's events
    kernel: FloatArray  # (n_i, U, K) G[k, u, j]
    counts_before: FloatArray  # (n_i, K) number of earlier type-j events


def target_sums(stream: EventStream, n_types: int, beta: FloatArray) -> list[TargetSums]:
    """`TargetSums` for each target type 0 .. n_types-1 under (U, K, K) decays."""
    require_types_within(stream, n_types)
    one_hot = np.stack([(stream.types == j).astype(np.float64) for j in range(n_types)], axis=1)
    counts_before = np.cumsum(one_hot, axis=0) - one_hot
    by_source_and_decay: dict[tuple[int, float], FloatArray] = {}
    targets = []
    for i in range(n_types):
        at_i = stream.types == i
        kernel = np.empty((int(at_i.sum()), beta.shape[0], n_types))
        for u in range(beta.shape[0]):
            for j in range(n_types):
                key = (j, float(beta[u, i, j]))
                if key not in by_source_and_decay:
                    by_source_and_decay[key] = _kernel_sum(stream.times, one_hot[:, j], key[1])
                kernel[:, u, j] = by_source_and_decay[key][at_i]
        targets.append(TargetSums(stream.times[at_i], kernel, counts_before[at_i]))
    return targets


def compensator_tails(stream: EventStream, n_types: int, beta: FloatArray) -> FloatArray:
    """tails[u, i, j] = sum over type-j events s of (1 - exp(-beta_uij (T - s))) / beta_uij.

    alpha_uij * tails[u, i, j] is component u of type j's total contribution
    to type i's compensator over [0, T].
    """
    tails = np.zeros(beta.shape)
    for j in range(n_types):
        remaining = stream.horizon - stream.times[stream.types == j]
        for u in range(beta.shape[0]):
            decay = beta[u, :, j]
            tails[u, :, j] = (1.0 - np.exp(-np.outer(decay, remaining))).sum(axis=1) / decay
    return tails


def require_types_within(stream: EventStream, n_types: int) -> None:
    if len(stream) and int(stream.types.max()) >= n_types:
        raise ValueError(f"stream has event types outside 0..{n_types - 1}")


def log_likelihood(stream: EventStream, params: HawkesParams) -> float:
    """Exact log-likelihood of `stream` under `params`.

    L = sum_k log lambda_{type_k}(t_k) - sum_i integral_0^horizon lambda_i(s) ds

    Intensities at events come from the vectorized kernel sums; the
    compensator (second term) is closed-form for exponential kernels.
    """
    total = 0.0
    for i, target in enumerate(target_sums(stream, params.n_types, params.beta)):
        intensity = params.mu[i] + np.einsum("nuj,uj->n", target.kernel, params.alpha[:, i, :])
        if np.any(intensity <= 0):
            return float("-inf")
        total += float(np.log(intensity).sum())
    tails = compensator_tails(stream, params.n_types, params.beta)
    compensator = params.mu.sum() * stream.horizon + float((params.alpha * tails).sum())
    return float(total - compensator)


class OnlineHawkes:
    """Ogata thinning one event at a time, for simulators that interleave the
    background flow with other activity.

    Between events every exponential component decays, so the current total
    intensity bounds the intensity until the next event. Draw a candidate at
    that rate; accept it as a type-k event with probability
    lambda_k(candidate) / bound, otherwise keep the candidate time (where the
    bound is tighter) and try again.

    A market simulator needs the next background event *before a deadline*
    (the next agent action), and needs agent orders to excite the flow as
    background events of the same type would. A candidate beyond the
    deadline is kept for the next call, since its bound stays valid while
    intensity only decays, so stopping at deadlines never changes the
    realization: stepping in increments reproduces one uninterrupted run
    exactly. An external `excite` raises intensity above that bound, so it
    discards the candidate; re-proposing from there is still exact, because
    thinning proposals are memoryless. `simulate` is this loop run to the
    horizon.
    """

    def __init__(self, params: HawkesParams, rng: np.random.Generator) -> None:
        params._require_stationary()
        self.params = params
        self.rng = rng
        self.now = 0.0
        self._excitation = np.zeros(params.alpha.shape)
        self._candidate: tuple[float, float] | None = None  # (time, upper bound it was drawn under)

    def intensity(self) -> FloatArray:
        """Current intensity of each type."""
        intensity: FloatArray = self.params.mu + self._excitation.sum(axis=(0, 2))
        return intensity

    def advance_to(self, t: float) -> None:
        if t < self.now:
            raise ValueError(f"cannot move back in time from {self.now} to {t}")
        if self._candidate is not None and t > self._candidate[0]:
            raise ValueError(f"advancing to {t} would skip the pending candidate at {self._candidate[0]}")
        self._excitation = _decayed(self._excitation, self.params.beta, t - self.now)
        self.now = t

    def excite(self, kind: int) -> None:
        """Register an event of `kind` at the current time (e.g. an agent's)."""
        self._excitation[:, :, kind] += self.params.alpha[:, :, kind]
        self._candidate = None  # its bound no longer covers the raised intensity

    def next_event(self, until: float) -> tuple[float, int] | None:
        """The next background event before `until`, already registered, or
        None after advancing the clock to `until`."""
        while True:
            if self._candidate is None:
                upper_bound = float(self.intensity().sum())
                self._candidate = (self.now + self.rng.exponential(1.0 / upper_bound), upper_bound)
            candidate, upper_bound = self._candidate
            if candidate >= until:
                self.advance_to(until)
                return None
            self._candidate = None
            self.advance_to(candidate)
            intensity = self.intensity()
            total = float(intensity.sum())
            if self.rng.uniform() * upper_bound <= total:
                kind = int(self.rng.choice(self.params.n_types, p=intensity / total))
                self.excite(kind)
                return self.now, kind

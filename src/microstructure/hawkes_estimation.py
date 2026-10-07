"""Maximum-likelihood estimation and goodness of fit for exponential Hawkes
processes.

With the decays fixed, the log-likelihood is concave in (mu, alpha) and
separates by target type: target i's terms involve only (mu_i, alpha_.i.)
and the events of type i. Writing x_k = (1, G[k, :, :]) for the kernel sums
at the k-th type-i event (one column per component and source type) and
b = (T, tails[:, i, :]) for the compensator coefficients, target i
contributes

    f(theta) = sum_k log(x_k . theta) - b . theta,     theta = (mu_i, alpha_i.) >= 0

It is solved in scale-free coordinates u_c = theta_c * b_c, each
coefficient's share of the compensator: with z_kc = x_kc / b_c the objective
becomes sum_k log(z_k . u) - sum_c u_c, and its gradient z^T (1/lambda) - 1 is
dimensionless. On real flow the raw columns span about seven orders of
magnitude, and L-BFGS-B in theta exhausted its evaluations with gradients of
~300 left (compensators 1-10% short of the event counts). In u it converges
in about a hundred iterations. A few EM steps (the multiplicative update
u_c <- u_c * sum_k z_kc / lambda_k, which never decreases the likelihood and
keeps u positive) give it a feasible start. At the optimum the KKT
conditions read: gradient 0 where u_c > 0 and <= 0 where u_c = 0, and the
fitted compensator equals the event count. `HawkesFit.kkt_residual` reports
the largest violation. The observed information X^T diag(1/lambda^2) X gives
standard errors.

Decays are therefore fixed, never jointly optimized with alpha (that problem
is non-convex and poorly identified). Two ways to choose them:

- `fit(stream, K, decay=[d_1, ..., d_U])`: a log-spaced grid of components
  shared by every pair (the `tick` library's HawkesSumExpKern), which is how
  multi-timescale real flow is fitted;
- `fit_decay`: one shared decay chosen by maximizing the profile likelihood
  (`tick`'s HawkesExpKern with a decay search).

Goodness of fit uses the time-rescaling theorem (Brown et al. 2002): under
the true model, compensator increments between consecutive events of each
type are i.i.d. Exp(1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import kstest

from microstructure.hawkes import (
    EventStream,
    FloatArray,
    HawkesParams,
    compensator_tails,
    log_likelihood,
    require_types_within,
    target_sums,
)


@dataclass(frozen=True)
class HawkesFit:
    """A fitted model with standard errors and information criteria.

    `decay_se` is None when the decay was fixed rather than estimated.
    Standard errors of parameters estimated at their zero bound come from the
    same information matrix and are only indicative there. `kkt_residual` is
    the largest violation of the optimality conditions in scale-free units
    (about 1e-7 for a converged fit).
    """

    params: HawkesParams
    mu_se: FloatArray
    alpha_se: FloatArray
    decay_se: float | None
    log_likelihood: float
    n_params: int
    kkt_residual: float = 0.0

    @property
    def aic(self) -> float:
        return 2.0 * self.n_params - 2.0 * self.log_likelihood


@dataclass(frozen=True)
class KSTest:
    statistic: float
    pvalue: float


def _objective(theta: FloatArray, design: FloatArray, linear: FloatArray) -> tuple[float, FloatArray]:
    """One target's log-likelihood sum(log(design @ theta)) - linear @ theta
    and its gradient; -inf where an intensity is not positive."""
    intensity = design @ theta
    if np.any(intensity <= 0):
        return float("-inf"), np.zeros_like(theta)
    gradient: FloatArray = design.T @ (1.0 / intensity) - linear
    return float(np.log(intensity).sum() - linear @ theta), gradient


_EM_WARM_START_STEPS = 50


def _maximize_target(design: FloatArray, linear: FloatArray) -> tuple[FloatArray, FloatArray, float, float]:
    """Maximize `_objective` over theta >= 0, with theta[0] (the background
    rate) kept strictly positive.

    Returns (theta, standard errors, maximum, KKT residual).

    A coefficient whose source type has no events has a zero column and a
    zero compensator coefficient: it affects nothing, so it is fixed at zero
    and left out of the optimization.
    """
    theta = np.zeros(design.shape[1])
    std_errors = np.zeros(design.shape[1])
    identified = linear > 0
    design, linear = design[:, identified], linear[identified]
    n_events, n_params = design.shape
    scaled = design / linear  # z: column c in units of its compensator share

    def negative(u: FloatArray) -> tuple[float, FloatArray]:
        intensity = scaled @ u
        if np.any(intensity <= 0):
            return float("inf"), np.zeros_like(u)
        gradient: FloatArray = scaled.T @ (1.0 / intensity) - 1.0
        return -float(np.log(intensity).sum() - u.sum()), -gradient

    u = np.full(n_params, n_events / n_params)
    for _ in range(_EM_WARM_START_STEPS):
        u *= scaled.T @ (1.0 / (scaled @ u))
    result = minimize(
        negative,
        u,
        jac=True,
        method="L-BFGS-B",
        bounds=[(1e-12 * n_events, None)] + [(0.0, None)] * (n_params - 1),
        options={"maxiter": 100_000, "maxfun": 200_000, "ftol": 1e-15, "gtol": 1e-10},
    )
    u = np.asarray(result.x, dtype=np.float64)
    _, gradient = negative(u)
    gradient = -gradient
    active = u > 1e-9 * n_events
    kkt = max(float(np.abs(gradient[active]).max(initial=0.0)), float(gradient[~active].max(initial=0.0)))

    theta[identified] = u / linear
    intensity = design @ theta[identified]
    information = design.T @ (design / intensity[:, None] ** 2)
    std_errors[identified] = np.sqrt(np.clip(np.diag(np.linalg.pinv(information)), 0.0, None))
    return theta, std_errors, -float(result.fun), kkt


def _decay_grid(decay: float | Sequence[float], n_types: int) -> FloatArray:
    """(U, K, K) decays: component u has decay[u] for every pair."""
    decays = np.atleast_1d(np.asarray(decay, dtype=np.float64))
    if decays.ndim != 1 or decays.size == 0 or np.any(decays <= 0):
        raise ValueError(f"decay must be a positive number or non-empty list of them, got {decay!r}")
    if np.unique(decays).size != decays.size:
        raise ValueError(f"decays must be distinct, got {decay!r}")
    grid: FloatArray = np.broadcast_to(decays[:, None, None], (decays.size, n_types, n_types)).copy()
    return grid


def _fit_with_decay(stream: EventStream, n_types: int, beta: FloatArray) -> HawkesFit:
    n_components = beta.shape[0]
    mu, mu_se = np.zeros(n_types), np.zeros(n_types)
    alpha, alpha_se = np.zeros(beta.shape), np.zeros(beta.shape)
    tails = compensator_tails(stream, n_types, beta)
    total, kkt = 0.0, 0.0
    for i, target in enumerate(target_sums(stream, n_types, beta)):
        n_i = target.times.shape[0]
        if n_i == 0:
            continue  # no events of type i: mu_i = alpha_.i. = 0 is the maximizer
        design = np.hstack([np.ones((n_i, 1)), target.kernel.reshape(n_i, -1)])
        linear = np.concatenate([[stream.horizon], tails[:, i, :].ravel()])
        theta, std_errors, value, residual = _maximize_target(design, linear)
        mu[i], alpha[:, i, :] = theta[0], theta[1:].reshape(n_components, n_types)
        mu_se[i], alpha_se[:, i, :] = std_errors[0], std_errors[1:].reshape(n_components, n_types)
        total += value
        kkt = max(kkt, residual)
    n_params = n_types + n_components * n_types**2
    return HawkesFit(HawkesParams(mu, alpha, beta), mu_se, alpha_se, None, total, n_params, kkt)


def fit(stream: EventStream, n_types: int, decay: float | Sequence[float]) -> HawkesFit:
    """Maximum-likelihood (mu, alpha) with fixed decays.

    A single `decay` gives one exponential per kernel; a list gives one
    component per decay, shared by every (i, j) pair.
    """
    require_types_within(stream, n_types)
    return _fit_with_decay(stream, n_types, _decay_grid(decay, n_types))


def fit_decay(stream: EventStream, n_types: int, bounds: tuple[float, float]) -> HawkesFit:
    """Maximum likelihood over (mu, alpha) and one shared decay within `bounds`.

    The decay maximizes the profile likelihood (a bounded search on log
    decay); its standard error comes from the profile's curvature.
    """
    require_types_within(stream, n_types)
    low, high = bounds
    if not 0 < low < high:
        raise ValueError(f"decay bounds must satisfy 0 < low < high, got {bounds!r}")

    def profile(decay: float) -> float:
        return _fit_with_decay(stream, n_types, _decay_grid(decay, n_types)).log_likelihood

    search = minimize_scalar(
        lambda log_decay: -profile(float(np.exp(log_decay))),
        bounds=(np.log(low), np.log(high)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    decay = float(np.exp(search.x))
    best = _fit_with_decay(stream, n_types, _decay_grid(decay, n_types))
    step = 1e-3 * decay
    curvature = (profile(decay + step) - 2.0 * best.log_likelihood + profile(decay - step)) / step**2
    decay_se = float(1.0 / np.sqrt(-curvature)) if curvature < 0 else float("nan")
    return HawkesFit(
        best.params,
        best.mu_se,
        best.alpha_se,
        decay_se,
        best.log_likelihood,
        best.n_params + 1,
        best.kkt_residual,
    )


def fit_poisson(stream: EventStream, n_types: int) -> HawkesFit:
    """The homogeneous Poisson null: mu_i = n_i / T and no excitation."""
    require_types_within(stream, n_types)
    counts = stream.counts(n_types).astype(np.float64)
    params = HawkesParams(counts / stream.horizon, np.zeros((n_types, n_types)), np.ones((n_types, n_types)))
    return HawkesFit(
        params=params,
        mu_se=np.sqrt(counts) / stream.horizon,
        alpha_se=np.zeros((n_types, n_types)),
        decay_se=None,
        log_likelihood=log_likelihood(stream, params),
        n_params=n_types,
    )


def rescaled_residuals(stream: EventStream, params: HawkesParams) -> list[FloatArray]:
    """Per type, the compensator increments between consecutive events.

    The compensator of type i at time t is
    mu_i t + sum_u sum_j (alpha_uij / beta_uij) (N_j(t) - G_uij(t)), where N_j
    counts earlier type-j events.
    """
    residuals = []
    weights = params.alpha / params.beta
    for i, target in enumerate(target_sums(stream, params.n_types, params.beta)):
        not_yet_decayed = target.counts_before[:, np.newaxis, :] - target.kernel
        excited = np.einsum("nuj,uj->n", not_yet_decayed, weights[:, i, :])
        compensator = params.mu[i] * target.times + excited
        residuals.append(np.diff(compensator, prepend=0.0))
    return residuals


def ks_exponential(residuals: list[FloatArray]) -> list[KSTest]:
    """Kolmogorov-Smirnov test of each residual series against Exp(1).

    A series with fewer than two residuals gets NaN statistics.
    """
    tests = []
    for series in residuals:
        if series.shape[0] < 2:
            tests.append(KSTest(float("nan"), float("nan")))
            continue
        result = kstest(series, "expon")
        tests.append(KSTest(float(result.statistic), float(result.pvalue)))
    return tests

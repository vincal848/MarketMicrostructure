"""Multivariate Hawkes process with exponential kernels.

The intensity of event type i at time t, given the history of all K types:

    lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} alpha_ij * exp(-beta_ij (t - t_k^j))

mu is the (K,) background rate, alpha and beta are (K, K) excitation and
decay matrices: a type-j event raises type-i's intensity by alpha_ij
immediately, decaying at rate beta_ij. See docs/DESIGN.md for how the K
dimensions map onto order-book event types.

Self- and cross-excitation can make the process explosive: the branching
matrix n_ij = alpha_ij / beta_ij is the expected number of type-i children
directly triggered by one type-j event, and the process is stationary iff
its spectral radius is < 1 (Hawkes & Oakes 1974). `simulate` refuses to run
otherwise, since the expected event count diverges.
"""

import numpy as np


def spectral_radius(alpha, beta):
    """Spectral radius of the branching matrix n_ij = alpha_ij / beta_ij.

    < 1 is necessary and sufficient for stationarity.
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    branching = alpha / beta
    eigvals = np.linalg.eigvals(branching)
    return float(np.max(np.abs(eigvals)))


def stationary_intensity(mu, alpha, beta):
    """Long-run average intensity per type, (I - n)^-1 mu with n = alpha/beta.

    Only defined when spectral_radius(alpha, beta) < 1.
    """
    mu = np.asarray(mu, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    k = len(mu)
    branching = alpha / beta
    rho = spectral_radius(alpha, beta)
    if rho >= 1:
        raise ValueError(
            "branching matrix spectral radius %.4f >= 1: no stationary "
            "intensity exists" % rho)
    return np.linalg.solve(np.eye(k) - branching, mu)


def simulate(mu, alpha, beta, T, seed=None):
    """Simulate one realization on [0, T] by Ogata's (1981) thinning.

    Returns a list of (time, type) pairs, type in range(K), sorted by time.

    Because the kernel is a decaying exponential, the total intensity is
    always highest right after the last event and falls monotonically until
    the next one. That makes the current intensity a valid (local) upper
    bound for a thinning proposal: draw a candidate from a Poisson process
    at that rate, then accept it as a real event of type k with probability
    lambda_k(candidate) / (upper bound), otherwise keep the candidate time as
    the new, tighter bound and try again.
    """
    mu = np.asarray(mu, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    k_types = len(mu)
    rho = spectral_radius(alpha, beta)
    if rho >= 1:
        raise ValueError(
            "branching matrix spectral radius %.4f >= 1: refusing to "
            "simulate a non-stationary process" % rho)

    rng = np.random.default_rng(seed)
    # R[i, j]: contribution of type j's past events to type i's intensity,
    # decayed to `last_t`. lambda(t) = mu + R(t).sum(axis=1).
    R = np.zeros((k_types, k_types))
    last_t = 0.0
    t = 0.0
    events = []

    while t < T:
        decayed = R * np.exp(-beta * (t - last_t))
        upper_bound = (mu + decayed.sum(axis=1)).sum()
        if upper_bound <= 0:
            break
        t = t + rng.exponential(1.0 / upper_bound)
        if t >= T:
            break
        decayed_at_t = R * np.exp(-beta * (t - last_t))
        lam_at_t = mu + decayed_at_t.sum(axis=1)
        if rng.uniform() <= lam_at_t.sum() / upper_bound:
            k = rng.choice(k_types, p=lam_at_t / lam_at_t.sum())
            events.append((t, int(k)))
            R = decayed_at_t
            R[:, k] += alpha[:, k]
            last_t = t
        # Rejected: t has still advanced, so the next loop iteration starts
        # its upper bound from the (lower) intensity at the rejected time.

    return events


def log_likelihood(times, types, mu, alpha, beta, T):
    """Exact log-likelihood of one realization under candidate parameters.

    L = sum_k log(lambda_{type_k}(t_k)) - sum_i integral_0^T lambda_i(s) ds

    The sum term uses the standard recursive evaluation for exponential
    kernels (Ozaki 1979) rather than re-summing over all past events at
    every point, so it costs O(n_events * K^2). The compensator (the second
    term) has a closed form for exponential kernels.
    """
    times = np.asarray(times, dtype=float)
    types = np.asarray(types, dtype=int)
    if len(times) != len(types):
        raise ValueError("times and types must have the same length")
    mu = np.asarray(mu, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    k_types = len(mu)

    order = np.argsort(times)
    times = times[order]
    types = types[order]

    R = np.zeros((k_types, k_types))
    last_t = 0.0
    loglik = 0.0
    for t_k, k in zip(times, types):
        decayed = R * np.exp(-beta * (t_k - last_t))
        lam = mu + decayed.sum(axis=1)
        if lam[k] <= 0:
            return -np.inf
        loglik += np.log(lam[k])
        R = decayed
        R[:, k] += alpha[:, k]
        last_t = t_k

    compensator = mu.sum() * T
    for j in range(k_types):
        t_j = times[types == j]
        if len(t_j) == 0:
            continue
        # sum_i alpha_ij / beta_ij * sum_{t_k^j < T} (1 - exp(-beta_ij (T - t_k^j)))
        decay_sum = np.sum(1.0 - np.exp(-beta[:, j][:, None] * (T - t_j)[None, :]), axis=1)
        compensator += np.sum(alpha[:, j] / beta[:, j] * decay_sum)

    return loglik - compensator


def fit_mle(times, types, n_types, x0=None):
    """Calibrate (mu, alpha, beta) to an observed event stream by MLE.

    TODO (M2): maximize log_likelihood numerically -- e.g. L-BFGS-B over a
    softplus/exp reparameterization to keep mu, alpha, beta positive, or a
    dedicated EM scheme (Veen & Schoenberg 2008, univariate case). M2's test
    is that this recovers the parameters used to simulate a long synthetic
    series within the known rate of statistical error.
    """
    raise NotImplementedError("MLE fitting is M2 work; see TODO above")

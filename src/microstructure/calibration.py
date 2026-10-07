"""Calibrate the six-type Hawkes model to classified flow, one window at a time.

Order flow is far from stationary over a day: activity follows a U shape,
heavy at the open and close. A Hawkes process with a constant background
rate can only describe a stretch where that rate is roughly flat. So
calibration fits short windows (30 minutes by default) inside the continuous
session, excluding the first and last 30 minutes, and reports how stable the
fit is across windows instead of pretending one fit describes the day.

Each window is fitted twice, as Hawkes (one shared decay, profiled) and as
homogeneous Poisson. For each fit the report gives:
- the AIC improvement of Hawkes over Poisson;
- the branching ratio (spectral radius of alpha / beta), the share of events
  triggered endogenously by earlier events;
- a per-type Kolmogorov-Smirnov test of time-rescaled residuals against
  Exp(1).

With hundreds of thousands of events per window, KS has power to reject
any misspecification however small, so its p-values are expected to be tiny
on real data. The statistic itself (the largest gap between the empirical
and Exp(1) CDFs) is the informative number, and Hawkes should shrink it
relative to Poisson.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from microstructure.flow import N_FLOW_TYPES, ClassifiedFlow, FlowType
from microstructure.hawkes import IntArray
from microstructure.hawkes_estimation import (
    HawkesFit,
    KSTest,
    fit_decay,
    fit_poisson,
    ks_exponential,
    rescaled_residuals,
)

NS_PER_MINUTE = 60 * 1_000_000_000


@dataclass(frozen=True)
class WindowFit:
    start_ns: int
    end_ns: int
    counts: IntArray
    hawkes: HawkesFit
    poisson: HawkesFit
    hawkes_ks: list[KSTest]
    poisson_ks: list[KSTest]

    @property
    def n_events(self) -> int:
        return int(self.counts.sum())

    @property
    def branching_ratio(self) -> float:
        return self.hawkes.params.spectral_radius()

    @property
    def aic_improvement(self) -> float:
        """Poisson AIC minus Hawkes AIC: positive when Hawkes fits better."""
        return self.poisson.aic - self.hawkes.aic

    def summary(self) -> dict[str, Any]:
        params = self.hawkes.params
        return {
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "types": [kind.name for kind in FlowType],
            "counts": [int(c) for c in self.counts],
            "decay": float(params.beta[0, 0]),
            "decay_se": self.hawkes.decay_se,
            "branching_ratio": self.branching_ratio,
            "aic_improvement": self.aic_improvement,
            "mu": params.mu.tolist(),
            "mu_se": self.hawkes.mu_se.tolist(),
            "alpha": params.alpha.tolist(),
            "alpha_se": self.hawkes.alpha_se.tolist(),
            "ks": {
                "hawkes_statistic": [t.statistic for t in self.hawkes_ks],
                "poisson_statistic": [t.statistic for t in self.poisson_ks],
                "hawkes_pvalue": [t.pvalue for t in self.hawkes_ks],
            },
        }


def session_windows(start_ns: int, end_ns: int, minutes: int) -> list[tuple[int, int]]:
    """Consecutive [start, end) windows of `minutes` length covering
    [start_ns, end_ns); a trailing partial window is dropped."""
    step = minutes * NS_PER_MINUTE
    return [(t, t + step) for t in range(start_ns, end_ns - step + 1, step)]


def calibrate_window(
    flow: ClassifiedFlow, start_ns: int, end_ns: int, decay_bounds: tuple[float, float]
) -> WindowFit:
    """Fit Hawkes and Poisson to the flow in [start_ns, end_ns)."""
    stream = flow.to_stream(start_ns, end_ns)
    hawkes = fit_decay(stream, N_FLOW_TYPES, decay_bounds)
    poisson = fit_poisson(stream, N_FLOW_TYPES)
    return WindowFit(
        start_ns=start_ns,
        end_ns=end_ns,
        counts=stream.counts(N_FLOW_TYPES),
        hawkes=hawkes,
        poisson=poisson,
        hawkes_ks=ks_exponential(rescaled_residuals(stream, hawkes.params)),
        poisson_ks=ks_exponential(rescaled_residuals(stream, poisson.params)),
    )

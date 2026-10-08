# M5 follow-up: can the DQN beat both baselines?

Declared before any variant was run (this file is committed first).

## Why

The M5 run (`docs/RESULTS.md`) shows the DQN beating Avellaneda-Stoikov
(AS) but tied with the 1-tick fixed spread (paired difference −10.8
[−116.5, +99.8] on test seeds 9000–9029). Those seeds have been looked at, so
they are not used again for anything below.

## Protocol

- **Market, costs, fees, agents**: exactly `configs/spy_20190130_1100.toml`
  (0.5 ms latency, maker rebate $0.0020, taker fee $0.0030, 1 s attribution,
  AS calibrated on seeds 0–3). Only the `[rl]` table and the env features
  change between variants.
- **Seeds** (all disjoint, enforced by the config loader):
  calibration 0–3, DQN training 1000–1999, checkpoint selection
  (validation) 5000–5009, **tuning 6000–6029**, **held-out 20000–20059**.
- **Tuning** compares variants: each is trained once (DQN seed 0) and run
  against all baselines on the tuning seeds. Nothing is ever run on the
  held-out seeds during tuning.
- **Selection rule**: the variant with the largest
  `min(mean paired PnL diff vs fixed, mean paired PnL diff vs AS)` on the
  tuning seeds. If that minimum is not positive, nothing is promoted.
- **Held-out evaluation**: the selected variant only, once, with no retuning
  afterwards, whatever the outcome.
- **Budget**: at most 6 variants (V1–V6 below), plus the unchanged config
  re-run as the reference V0 (not a variant). Every one actually run is
  counted in the PR and listed in the results table, including failures.
- **Success criterion**: on the held-out seeds, the 95% paired-bootstrap CI
  of (DQN − fixed spread) PnL per episode excludes zero on the positive side,
  AND so does (DQN − AS). Computed by `experiment.paired_pnl_report`
  (`beats`), written to `manifest.json` under `dqn_paired_pnl`. PnL is the
  repo's existing mark-to-market PnL with fees.
- **Null / planted checks**: `tests/test_experiment.py` runs the paired
  report on a tied pair (must not beat) and a planted +50 edge (must beat).
- **If nothing clears the bar**: the README says the DQN does not beat the
  fixed spread yet, with the numbers.

## Variants

| id | change | rationale |
|---|---|---|
| V0 | none (reference) | reproduces the M5 setting on the tuning seeds |
| V1 | `inventory_penalty` 0.002 → 0.0086 | matches AS's risk term 0.5·γσ²q² per 1 s step (γ=1e-4, σ=1.311 ticks/√s, q=100) in the reward's units |
| V2 | extra observable flow features: signed trade-volume imbalance and mid move over the last 5 s and 30 s | the 1 s features see too little of the clustered flow; no latent Hawkes intensity (a real participant cannot read it, and the baselines do not get it) |
| V3 | 600 training episodes, epsilon decay 90,000 steps | more training; validation PnL was still volatile at 200 |
| V4 | offsets (1, 2, 3, 4): never quote at the touch | the fixed spread quotes 1 tick behind the touch, AS quotes at it and bleeds adverse selection; (1,1) is the fixed-spread action, so the agent can at worst match it |
| V5 | the best of V1–V4 by the selection rule, plus every other single change whose mean diff vs V0 on the tuning seeds is positive | combination, spent only after V1–V4 |
| V6 | reserved, may remain unused | |

Caveats stated up front: one DQN training seed per variant (DQN validation
PnL is noisy, so tuning-seed differences between variants can be luck);
training is deterministic, so the held-out run retrains the selected variant
and checks the policy file is identical to the tuned one.

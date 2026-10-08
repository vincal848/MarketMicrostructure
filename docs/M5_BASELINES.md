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

## Results

Dollars per 5-minute episode (run files are in 1/10000-dollar units).
Paired differences are DQN minus baseline, 95% bootstrap CI.

**Tuning seeds 6000–6029** (fixed spread +2.8, AS −630.6 on these seeds):

| id | DQN PnL | DQN − fixed | DQN − AS | selection score (min of the two means) |
|---|---|---|---|---|
| V0 | −14.9 | −17.8 [−87.0, +66.4] | +615.7 [+328.0, +984.3] | −17.8 |
| V1 λ=0.0086 | +52.4 | +49.6 [−12.6, +118.6] | +683.0 [+387.4, +1,066.5] | **+49.6** |
| V2 flow windows | +50.8 | +48.0 [−21.1, +122.5] | +681.4 [+396.7, +1,065.5] | +48.0 |
| V3 600 episodes | −9.9 | −12.7 [−82.5, +63.1] | +620.8 [+318.0, +1,007.8] | −12.7 |
| V4 offsets 1–4 | +6.2 | +3.4 [−73.0, +80.8] | +636.8 [+329.6, +1,036.3] | +3.4 |
| V5 all four combined | +20.1 | +17.2 [−45.0, +77.5] | +650.7 [+362.1, +1,014.9] | +17.2 |

Variants run: V1–V5, five of the budget of six (V6 unused), plus the V0
reference. No tuning-seed CI against the fixed spread excluded zero. V1 won
the selection rule and was promoted.

**Held-out seeds 20000–20059, V1 only, run once** (fixed spread +20.0, AS
−771.7, AS capped −778.5, DQN +12.0):

| comparison | paired PnL difference | CI excludes 0 (positive)? |
|---|---|---|
| DQN − fixed spread | −8.1 [−76.2, +64.4] | **no** |
| DQN − AS | +783.7 [+502.7, +1,087.6] | yes |
| DQN − AS capped | +790.5 [+510.3, +1,095.7] | yes |

**The success criterion is not met**: the DQN beats AS on held-out seeds
but does not beat the fixed spread. The tuning-seed gain of V1 over V0
(about +67 against the fixed spread) did not carry over; with one training
seed per variant and 10 validation seeds for checkpointing, it was
within noise.

Check that failed: retraining V1 for the held-out run did not reproduce the
tuned policy file (the tuning runs shared the machine under different torch
thread counts, so float summation order differed). The held-out policy is
therefore a second training realisation of V1, not the tuned weights.
Neither was ever fitted to the held-out seeds.

What to try next (not run, each would need new tuning and a new held-out
set, so the 20000–20059 seeds are now spent): several DQN training seeds
per variant and selecting by their mean; a larger validation set for
checkpoint selection (10 seeds gives noisy picks, see validation PnL
swinging between −0.3M and +1.3M across checkpoints); evaluating against
the fixed spread on the reward itself (e.g. rewarding PnL relative to the
fixed-spread policy on the same seed); an exact fixed-spread action with
uncapped inventory.

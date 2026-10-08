# M5 follow-up 2: residual reward, deviation actions, several training seeds

Declared before any code for it was written or run (this file is committed
first). Round 1 is `docs/M5_BASELINES.md`: its held-out seeds 20000–20059
are spent and are not used here.

## Protocol

- **Market, costs, fees, agents**: `configs/spy_20190130_1100.toml` as in
  round 1 (0.5 ms latency, maker rebate $0.0020, taker fee $0.0030, 1 s
  attribution, AS calibrated on seeds 0–3).
- **Seeds**, all disjoint: calibration 0–3, DQN training 1000–1999,
  checkpoint validation **5000–5029** (30 seeds, up from 10), **tuning
  7000–7029**, **held-out 30000–30059** (never used before).
- **Reproducibility**: `train_dqn` pins `torch.set_num_threads(1)`. The
  held-out run does not retrain; it loads the saved `policy.pt` of the
  promoted policy.
- **Variants** (budget: 4, nothing beyond them). All share round 1's V1
  base: inventory penalty 0.0086, 200 episodes, 64x64 network, default
  offsets unless stated.
  - **W1** plain base. This is the control that applies procedure (c)
    alone.
  - **W2** residual reward: reward = (agent MTM change − MTM change of a
    shadow 1-tick fixed-spread agent run on the same seed) / (tick x size)
    − inventory penalty. The agent is paid only for what it adds over the
    fixed spread.
  - **W3** deviation actions: the action grid is (bid, ask) deviations of
    (−1, 0, +1, +2) ticks from the fixed-spread quote (post-only), and the
    Q-network is initialised so the greedy policy is the zero deviation,
    i.e. the fixed spread. Plain reward.
  - **W4** W2 plus the 5 s / 30 s flow features (round 1's V2).
- **Procedure (c) for every variant**: 3 DQN training seeds (0, 1, 2).
  Each trained policy is scored on the tuning seeds by
  `s = min(mean paired PnL diff vs fixed spread, mean diff vs AS)`. The
  variant's score is the **median** of its three `s`; its promoted policy
  is the training seed that has that median.
- **Selection**: the variant with the highest score is promoted. Its
  promoted policy is evaluated once on the held-out seeds, even if its
  tuning score is not positive (so the report rests on fresh data), with no
  retuning afterwards.
- **Success**: on 30000–30059, the paired 95% bootstrap CI of (DQN − fixed
  spread) AND of (DQN − AS) PnL per episode both exclude 0 on the positive
  side (`paired_pnl_report`, `beats`).
- **Null / planted checks**: unit tests run the paired report on a tied pair
  and a planted edge (round 1); the residual reward is tested to equal the
  plain reward minus exactly the shadow's MTM change.
- **If nothing clears**: README says so with the numbers, and that the
  fixed spread is a strong baseline in this simulator. Stop there.

## Amendment, committed before any variant ran

I had planned a test that the residual reward is zero when the agent
quotes the fixed-spread quote. It is not: on the unit-test scenario (60 s
episodes, 40 seeds) the residual of an agent playing exactly the fixed quote
averages −13.7 ± 5.5 tick-share units with a standard deviation of about 35,
because the shadow is a separate simulation in which the agent's orders
change the book differently (as in `evaluation`'s pairing). So the shadow
is a noisy, slightly pessimistic counterfactual, not a control variate. W2
and W4 are run as declared; the accounting-identity test replaces the
zero-residual test.

## Amendment 2: W3's action grid (compute, made after W1, W2, W4 finished)

The first W3 runs (deviations −1, 0, +1, +2) did not finish in over an hour
and were killed with no result, no tuning score and no checkpoint seen. Cause:
`MarketSimulator._settle_fills` schedules an extra `_DECIDE` event on every
passive fill, and each `_DECIDE` reschedules itself, so an agent gets one
more perpetual decision chain per fill. A policy quoting inside the fixed
spread fills several times as often as the baselines, and the number of
decision events grows with it. This is an existing simulator quirk that
affects every agent (more fills, more requotes) and is not fixed here, since
fixing it would change the baselines' numbers. W3 is rerun with deviations
(0, +1, +2, +3): never tighter than the fixed spread, zero deviation as the
prior (action 0). It is the same W3 variant, so the budget stays at 4
variants. W1, W2 and W4 results were visible when I chose the new grid;
the choice was driven by run time only.

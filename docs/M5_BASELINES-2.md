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
  and a planted edge (round 1); the residual shadow agent is tested to give
  zero residual reward when the agent quotes the fixed-spread quote.
- **If nothing clears**: README says so with the numbers, and that the
  fixed spread is a strong baseline in this simulator. Stop there.

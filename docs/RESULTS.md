# Results

Acceptance numbers measured on real data, with the command that produced
each one. Real data is not in the repository (see README > Data), so these
runs are local. Every number here is reproducible from the listed command
given the same input file.

## M1: exact order-level replay (Phase 1)

**Input.** Nasdaq TotalView-ITCH 5.0 sample day `01302019.NASDAQ_ITCH50.gz`
(4,764,426,091 bytes, MD5 `54b7afd0be7f925f7e9d67a23898827b`), symbol SPY.

**Command.**

```bash
microstructure replay-itch /path/to/01302019.NASDAQ_ITCH50.gz --symbol SPY \
    --out runs/m1_replay_spy_20190130.json
```

| Measure | Value |
|---|---|
| SPY order events replayed (full day, 04:00–20:00) | 3,065,003 |
| Visible executions audited (09:30–16:00) | 130,532 |
| Price-time priority violations | **0** |
| References to unknown order ids | **0** |
| Cancels/executions larger than the resting order | **0** |
| Displayed adds that would have crossed the book | **0** |
| Executions with an explicit price (crosses, not audited) | 4,637 |
| Hidden (non-displayed) executions | 9,041 |
| Book after the close | empty |
| Wall time for the whole file (all symbols decoded or skipped) | 230.5 s |

**Reading.** Every one of the 130,532 visible executions during continuous
trading hit the order the engine had first in line at the best price. So
the book reproduces Nasdaq's matching order exactly on a full day of a very
liquid symbol, and the depth it produces can be trusted as the base for
classification and calibration. The run took 230 s on one core. Most of
that is decompressing and skipping the other ~8,500 symbols' messages;
filtering one symbol costs one two-byte read per message.

**Acceptance test.** `MICROSTRUCTURE_ITCH=... pytest -m data` runs the same
replay and asserts every count above that should be zero is zero.

## M3: calibration to real flow (Phases 3, 3b and the Phase 2 fix)

**Inputs.** The ITCH day above (SPY), and Databento `XNAS.ITCH` MBP-10 for
SPY on 2025-11-11 (the owner's licensed copy). Session 10:00–15:30 New York
time, in eleven 30-minute windows. The six-type alphabet is MB, MS, LA, LI,
LD, C ([ARCHITECTURE.md](ARCHITECTURE.md)).

**Commands.**

```bash
microstructure calibrate-itch  01302019.NASDAQ_ITCH50.gz --symbol SPY \
    --out runs/m3_calibrate_itch_spy_20190130.json --marks-out runs/m3_marks_itch_spy_20190130.npz
microstructure calibrate-mbp10 xnas-itch-20251111.mbp-10.csv \
    --out runs/m3_calibrate_mbp10_spy_20251111.json --marks-out runs/m3_marks_mbp10_spy_20251111.npz
# the single-exponential baseline: add --single-decay
```

| | ITCH, 2019-01-30 | MBP-10, 2025-11-11 |
|---|---|---|
| Classified session events | 2,458,665 | 6,335,439 |
| Events per 30-minute window | 105,434 – 477,932 | 299,941 – 965,302 |
| Type shares MB / MS / LA / LI / LD / C | 1.1 / 1.1 / 25.6 / 2.3 / 22.7 / 47.3 % | 0.7 / 0.7 / 27.2 / 3.3 / 19.2 / 49.0 % |
| **Single exponential**: decay at the 5000/s search bound | 11 of 11 windows | 11 of 11 windows |
| **Decay grid** (1e5 … 1 /s): AIC gain over Poisson | 0.77M – 3.02M | 2.19M – 5.42M |
| Decay grid: AIC gain over the single exponential | +73k – +369k | +151k – +436k |
| Largest KKT residual (convergence check) | 1.7e-6 | 7.3e-7 |
| Branching ratio (endogeneity) | 0.908 – 0.962 (median 0.928) | 0.935 – 0.966 (median 0.957) |
| KS distance of residuals, median per type: Hawkes | 0.02 – 0.07 | 0.02 – 0.07 |
| KS distance of residuals, median per type: Poisson | 0.31 – 0.68 | 0.30 – 0.66 |

**Reading.**
- **Hawkes beats Poisson by AIC in every window of both eras, by a wide
  margin.** On the time-rescaling test, the worst-fitting type's residuals
  move 5 to 30 times closer to Exp(1).
- **KS p-values are not used to accept or reject.** With 10⁵–10⁶ events
  per window, KS rejects any model of real flow, so the statistic itself is
  what is compared.
- **About 93–96% of events are triggered by earlier events.** That is the
  near-critical reflexivity of Hardiman, Bercot & Bouchaud (2013) and
  Filimonov & Sornette (2012).
- **Flow is more endogenous and 2.6× busier in 2025 than in 2019, with a
  nearly unchanged event mix.**
- **Phase 3b was needed.** A single exponential cannot describe this flow:
  61% of gaps are under 100 µs while clustering persists for seconds, and
  its decay always ran to the bound.

**What the first numbers got wrong.** The first decay-grid fits (logged in
CHANGELOG, "Phase 2 fix") had not converged, and their compensators were
1–10% short of the event counts. On the 11:00 window, the converged fit's
stationary intensity is 131.4/s against 131.3/s realized; the unconverged
fit implied 74/s. Every number in the table is from converged fits.

## M3 validation: the simulated market against the real tape (Phase 4)

**Scenario.** The 11:00–11:30 ITCH window: its fitted decay-grid Hawkes
process, session mark samples with placements capped at 20 ticks
(`max_distance_ticks`), and the real book at 11:00 (`depth-itch`). Five
30-minute simulations (seeds 0–4), compared with the real tape over the
same window (`stylized-itch`).

```bash
microstructure simulate configs/spy_20190130_1100.toml --seeds 0:5 --horizon 1800 \
    --out runs/m3_simulate_spy_20190130_1100.json
microstructure stylized-itch 01302019.NASDAQ_ITCH50.gz --symbol SPY --start 11:00 --end 11:30 \
    --out runs/stylized_real_spy_20190130_1100.json
```

| Statistic | Real tape | Simulated (mean of 5) | Verdict |
|---|---|---|---|
| Event rate / stationary intensity, per type | — | 0.989 – 1.000 | **meets** the 5% acceptance |
| Trades | 8,920 | 9,183 | matches |
| Quote updates | 223,020 | 235,722 | matches |
| Mean inter-trade time | 0.202 s | 0.197 s | matches |
| Inter-trade time CV (burstiness) | 2.99 | 3.48 | close; simulated flow slightly burstier |
| Trade-sign autocorrelation, lags 1–5 | .607 .412 .294 .234 .192 | .657 .457 .340 .260 .208 | matches shape and decay |
| Spread = 1 / 2 / 3 ticks | 78.1 / 20.9 / 0.7 % | 98.7 / 0.5 / 0.2 % | **too tight** (TV distance 0.21) |
| Mean spread | 1.24 ticks | 1.04 ticks | too tight |
| Mid realized variance at 0.1 / 1 / 10 / 60 s | 12.9k / 12.5k / 12.4k / 8.5k | 49k / 40k / 24k / 30k | **2.5–4× too volatile** (1.6–2× in s.d.) |

**Reading.** What the Hawkes model controls (event timing, clustering,
the order-flow sign memory) the simulator reproduces: rates exact, trade
counts and durations within a few percent, sign autocorrelation matching at
every lag. What depends on *how the book reacts to its own state* is
off in the expected direction:

- **The spread is too often one tick.** Real liquidity providers let the
  spread widen to two ticks a fifth of the time.
- **The mid is too volatile.** Events arrive at the right rates whatever
  the queue sizes, so queues sometimes empty in bursts that real traders
  would have refilled.

This is the known limitation of state-independent Hawkes books, and the
reason queue-reactive intensities (Huang, Lehalle & Rosenbaum 2015) are the
roadmap's first extension. Three simulator defects found on the way here
(volume drift, unclosed gaps, a stub-quote reservoir) are fixed and logged.
Before those fixes, simulated variance was 4 (a frozen book), then 5×10⁷
(gaps that never closed).

## M4 and M5: market makers on the calibrated market (Phases 5 and 6)

**Experiment.** `configs/spy_20190130_1100.toml`, run directory
`runs/experiments/20261007T071910Z-spy-20190130-1100/` (manifest pins git
`88157e3`, clean; numpy 2.5.3, scipy 1.18.1, torch 2.14.1+cpu).

- Market: the validated 11:00 SPY scenario above, in 5-minute episodes.
- Order latency: 0.5 ms.
- Fees: maker rebate $0.0020/share, taker fee $0.0030/share.
- PnL attribution horizon: 1 s.
- Seeds, all disjoint: calibration 0–3, DQN training 1000–1999 (200
  episodes used), validation 5000–5009, test 9000–9029.
- Avellaneda-Stoikov is calibrated on the calibration seeds, in ticks:
  σ = 1.311 ticks/√s, κ = 2.664 /tick, γ = 1e-4 (a preference, not
  estimated).
- The DQN is a 64×64 Double DQN with 16 offset actions and inventory
  penalty λ = 0.002. The checkpoint with the best validation PnL
  (episode 160 of 200) is the one tested.

```bash
microstructure experiment configs/spy_20190130_1100.toml --out runs/experiments
```

Results per 5-minute episode, mean over the 30 test seeds, in dollars
(the run files report 1/10000-dollar units):

| Agent | PnL [95% CI] | Spread capture | Adverse selection (1 s) | Inventory PnL | Fees | Fills | Max \|inventory\| |
|---|---|---|---|---|---|---|---|
| Fixed spread, 1 tick | +21.7 [−72.6, +108.8] | +45.4 | −59.7 | +31.0 | −5.1 (rebate) | 92.0 | 1,032 |
| Avellaneda-Stoikov | **−946.3** [−1,329.8, −606.7] | +40.2 | **−526.7** | −471.5 | −11.7 | 150.7 | 320 |
| AS, inventory cap 500 | **−815.3** [−1,104.6, −546.7] | +40.3 | −504.5 | −362.9 | −11.8 | 150.3 | 316 |
| Double DQN | +10.9 [−40.7, +55.7] | +37.8 | **+35.5** | −64.3 | −1.9 | 66.1 | 519 |

Paired differences on the same test seeds (bootstrap 95% CI):

| Comparison | PnL difference per episode |
|---|---|
| DQN − Avellaneda-Stoikov | **+957.2** [+642.7, +1,325.4] |
| DQN − AS capped | **+826.2** [+583.2, +1,091.4] |
| DQN − fixed spread | −10.8 [−116.5, +99.8], not significant |

**M4 acceptance: met.** Every baseline traded in every one of the 30 test
seeds (minimum 5 fills for the fixed spread, 29 for AS). This was the
legacy project's failure. PnL is attributed for every run, and the
components sum exactly to mark-to-market PnL.

**M5 acceptance: met** (trained reproducibly from the config and seed,
compared on held-out seeds with CIs and attribution).

**Reading.**
- **The attribution says why AS loses.** Calibrated correctly, it quotes at
  the touch and trades the most (151 fills an episode). Its spread capture
  is positive, but its fills are followed by the price moving through them:
  −$527 of adverse selection within one second, and more afterwards in the
  inventory term. This is the failure the closed form ignores.
  Avellaneda-Stoikov assumes fills are uninformed, and in a market whose
  flow clusters (branching ratio 0.94 here), they are not.
- **The DQN learned to avoid toxic fills rather than to earn more spread.**
  It trades less than half as often as AS, and its adverse-selection term
  is *positive*: its fills tend to be followed by favourable moves. That is
  the kind of state-dependent quoting the observation (imbalance, recent
  signed flow, queue position) makes possible. It beats both AS variants
  decisively.
- **The DQN is not significantly better than the fixed spread.** Neither
  makes significant money: both CIs straddle zero. So the honest headline
  is that learning avoided the closed form's adverse selection, not that it
  found a profitable strategy.
- **The magnitudes are probably overstated.** The simulator's mid is
  2.5–4× too volatile at these horizons (see M3 validation), which inflates
  adverse selection for every agent, so the absolute losses should not be
  read literally. The paired comparisons are on identical simulated flow:
  event times come from a separate random stream, so agents cannot perturb
  them.
- **Validation PnL was volatile across checkpoints** (+0.05M, −0.26M,
  −0.17M, +1.00M, +0.79M at episodes 40–200). Selecting on validation seeds
  and testing on separate seeds is what keeps that from inflating the test
  result.

## Throughput (Phase 7)

`microstructure bench --out runs/bench.json` (seeded synthetic workloads,
one core, Windows 11, Python 3.13):

| Hot path | Throughput |
|---|---|
| Order book operations (add / cancel / market, 20 live levels) | 84,500 ops/s |
| Replay of normalized events | 610,000 events/s |
| Simulator (6-type Hawkes, background flow only) | 26,900 events/s, about 200× real time for SPY |

The real-data acceptance test (`MICROSTRUCTURE_ITCH=... pytest -m data`,
the full-day M1 replay) passes in 149 s.

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

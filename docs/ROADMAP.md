# Roadmap

A phased plan from the original scaffold to a calibrated, validated market
simulator with closed-form and learned market makers. [MOTIVATION.md](MOTIVATION.md)
says why; this file says what gets built, in what order, and how each phase
proves it works.

## Working rules (all phases)

- **Tests first.** Each phase starts by committing its acceptance tests
  (failing), then the implementation that makes them pass. The change log
  ([CHANGELOG.md](../CHANGELOG.md)) records both steps per phase.
- **Green gate.** `ruff check`, `ruff format --check`, `mypy --strict` and
  `pytest` pass in CI on Python 3.11–3.13 before `main` moves. Work happens on
  the `rebuild` branch; `main` is only fast-forwarded to a green commit.
- **Real data stays local.** Licensed or very large data (ITCH, LOBSTER,
  Databento) never enters git or CI. CI runs on small hand-made fixtures in
  the same formats. Tests that need real data are marked `@pytest.mark.data`
  and skip when the file is absent. Their results are recorded in
  [RESULTS.md](RESULTS.md) with the command that produced them.
- **Honest reporting.** Every phase reports its acceptance numbers whether or
  not they flatter the model. Where a target is missed, the gap and its cause
  are written down rather than the target being quietly moved.
- **Boring dependencies.** numpy, pandas and scipy for the core. PyTorch only
  as an optional `rl` extra. No framework is adopted where 100 lines of direct
  code will do.

## Data sources

| Source | Granularity | Role | Availability |
|---|---|---|---|
| Nasdaq TotalView-ITCH 5.0 sample day (2019-01-30) | Order level, all symbols, full day | Primary: exact replay (Phase 1), calibration (Phase 3) | Public download from `emi.nasdaq.com`, about 4.7 GB gz, stored outside the repo |
| LOBSTER message + orderbook files | Order level, one symbol | Same event model as ITCH; the adapter is verified exactly against LOBSTER's own snapshots | Account required; fixture-tested, run locally when files exist |
| Databento `XNAS.ITCH` MBP-10, SPY, 61 days (2025-11 to 2026-02) | Price level, top 10 | Cross-era check: does a 2019 calibration describe 2025 flow? (Phase 3) | The owner's licensed copy, local only |

The MBP-10 records do not carry order ids, and order replaces appear only as
an add at the new price. The level removed at the old price is visible only
in the attached snapshot (verified on 2025-11-11, record 30). So MBP data is
treated as *snapshots plus trades*, and level events are derived by diffing
consecutive snapshots. It cannot support exact order-level replay; ITCH can.

## Status

Every phase below is implemented, test-first; [CHANGELOG.md](../CHANGELOG.md)
logs each one as a tests commit followed by an implementation commit.
Real data forced two additions that were not in the original plan. Both are
recorded where they belong:

- **Phase 3b**: multi-timescale kernels. A single exponential's decay ran
  to its search bound on every real window.
- **The Phase 2 estimator fix**: scale-free coordinates. Real-flow fits had
  stopped short of the optimum.

---

## Phase 0: Foundation ✅

Typed package (`src/microstructure`), `pyproject.toml`, strict mypy, ruff, CI
lint and test jobs, regression tests for the input-validation defects found in
review, documentation set. Details in CHANGELOG 0.2.0.

## Phase 1: Order-level data and exact replay (milestone M1) ✅

**Goal.** Reconstruct a real order book message by message, and prove the
book engine enforces price-time priority on real flow.

**Deliverables**
- `events.py`: the normalized order-event model, shared by every source.
  Frozen dataclasses `AddOrder`, `CancelOrder` (partial), `DeleteOrder`,
  `ReplaceOrder`, `ExecuteOrder` and `HiddenTrade`, all with integer
  nanosecond timestamps and integer prices in 1/10000 dollars.
- `itch.py`: streaming Nasdaq TotalView-ITCH 5.0 decoder (binary,
  length-prefixed, gzip). It filters one symbol by stock-locate code and maps
  messages A/F/E/C/X/D/U/P onto `events.py`. A small writer for synthetic
  ITCH bytes is used by tests.
- `book.py` additions: `execute_order(order_id, qty)`, which fills a specific
  resting order as an exchange execution does, and `queue_ahead(order_id)`,
  the shares ahead of an order at its price.
- `replay.py`: drives `OrderBook` from an event stream and audits every
  execution. A visible execution must hit the order at the *front* of the
  *best* level; any other execution is a price-time priority violation.
- `lobster.py` additions: `to_events` (message types 1–5 to the event model),
  seeding the book from the first snapshot (LOBSTER windows open with orders
  already resting), and snapshot verification against the orderbook file.

**Tests written first**
- Synthetic ITCH byte streams round-trip through the decoder, covering every
  supported message type and the symbol filter.
- Replay of a hand-built event stream with known final depth; a deliberately
  out-of-priority execution is reported as a violation.
- LOBSTER fixture: replayed depth equals the orderbook file at every row,
  including a window that opens with seeded orders.
- `execute_order` and `queue_ahead` unit tests, including partial executions
  that keep queue position.

**Acceptance on real data (2019-01-30 ITCH, SPY, full day)**
- Zero references to unknown order ids.
- Zero price-time priority violations among visible executions.
- The book is never crossed during continuous trading (09:30–16:00).
- Replay throughput recorded (messages per second).

## Phase 2: Hawkes estimation (milestone M2) ✅

**Goal.** Fit a multivariate Hawkes process reliably, with uncertainty and a
goodness-of-fit test.

**Deliverables**
- `hawkes.fit(stream, decay)`: concave maximum likelihood over `(mu, alpha)`
  for a fixed decay matrix. Uses L-BFGS-B with an analytic gradient over
  precomputed kernel sums, so each likelihood evaluation is vectorized
  O(n·K²).
- `hawkes.fit_decay(stream, ...)`: profile likelihood over a shared decay
  `beta` (bounded scalar search with the inner concave fit).
- Standard errors from the observed Fisher information.
- `hawkes.rescaled_residuals(stream, params)` and a per-type
  Kolmogorov-Smirnov test of the residuals against Exp(1) (time-rescaling
  theorem).
- `hawkes.poisson_fit` and AIC, for comparing against the Poisson null.

**Tests written first**
- Parameter recovery on long simulated 1-D and 2-D streams: every fitted
  parameter lies within 4 standard errors of the truth.
- The analytic gradient matches finite differences.
- Residuals under the true parameters pass KS (p > 0.01); residuals of a
  Hawkes stream under a Poisson fit fail it.
- The fitted log-likelihood is at least the log-likelihood at the truth.

**Acceptance.** The recovery tests above, in CI, with a runtime budget.

**Added after real data.** On real SPY flow the optimizer stopped short of
the optimum, with fitted compensators 1–10% below the event counts. The
fitter now solves in compensator-share coordinates with an EM warm start,
and reports `kkt_residual`. The regression test is a six-type process with
tied replace events, which failed under the old optimizer.

## Phase 3: Event classification and calibration to real flow (milestone M3) ✅

**Goal.** Turn replayed order flow into the six-type event alphabet and fit it.

**Deliverables**
- `flow.py`: classifies replay output into MB, MS, LA, LI, LD, C (table in
  [ARCHITECTURE.md](ARCHITECTURE.md)), using the book state *before* each
  event:
  - Executions sharing one timestamp and aggressor side are merged into a
    single market order.
  - A replace counts as a cancel followed by an add.
  - Hidden executions count as market orders but have no book effect.
- `flow.marks`: empirical mark distributions used by the simulator (order
  sizes per type; placement distance in ticks for LD; the depth level hit by
  cancels).
- `calibration.py`: fits K = 6 on intraday windows (excluding the first and
  last 30 minutes), then reports the fitted parameters, branching ratio,
  per-type KS statistics, and AIC against Poisson.
- `databento.py`: MBP-10 snapshot-diff adapter producing the same six-type
  stream at level granularity, used to fit 2025 SPY windows.

**Tests written first**
- Classification of hand-built sequences: a sweep across three resting orders
  is one MB; adds at, inside and away from the touch; replace = C + add.
- The MBP-10 diff adapter on a fixture that includes the "replace shows only
  the add" quirk and records with side `N`.
- Calibration end to end on a synthetic stream with known parameters.

**Acceptance on real data**
- Hawkes beats Poisson by AIC on every window.
- Branching ratio and per-type KS statistics reported per window, with the
  fit's stability across windows.
- 2019 ITCH vs 2025 MBP-10: fitted rates and branching ratios compared and
  reported.

## Phase 3b: Multi-timescale kernels ✅ (added after Phase 3's first real fit)

With one exponential per kernel, the fitted decay hit its upper bound
(5000/s) in all 22 real windows. 61% of SPY inter-event gaps are under
100 µs and 98% are under 100 ms, a range no single exponential spans.
Kernels are now sums of exponentials on a fixed log-spaced decay grid
(`DecayGrid`, default 1e5 … 1 per second), which keeps the fit concave
(the approach of `tick`'s HawkesSumExpKern). Tests: two-scale recovery,
AIC preferring two scales over either one, and KS under the true model.

## Phase 4: Generative simulator (milestone M3, validation half) ✅

**Goal.** A discrete-event market that produces realistic flow and accepts
agent orders.

**Deliverables**
- `simulator.py`: event-driven engine with a single clock.
  - Background flow comes from the calibrated Hawkes process, simulated
    *online* by thinning, so agent market orders feed back into the
    intensities.
  - Each background event maps to a book operation, with marks sampled from
    Phase 3.
  - The book starts from a real snapshot.
  - Agents act through a typed `Agent` protocol. Their actions arrive after a
    configurable latency, and their fills are delivered as callbacks.
- `stylized.py`: spread distribution, trade-sign autocorrelation, signature
  plot (realized variance against sampling interval), inter-trade durations,
  and mid-price volatility. Computed identically for replayed real data and
  for simulated data.

**Tests written first**
- Invariants: the book is never crossed, background cancels never touch agent
  orders, an agent order reaches the book exactly `latency` after it was sent,
  and the same seed reproduces the run.
- Online thinning without agents reproduces the offline simulator's event
  rates.
- An agent market order raises background intensities by `alpha[:, type]`.

**Acceptance.** Simulated event rates within 5% of the calibrated stationary
intensities. The stylized-fact comparison against the replayed real day is
reported side by side in RESULTS.md, with discrepancies quantified (for
example the total-variation distance between spread distributions).

## Phase 5: Market makers and PnL attribution (milestone M4) ✅

**Goal.** Baseline market makers evaluated with an industry-style PnL split.

**Deliverables**
- `accounting.py`: a ledger tracking cash, inventory, maker/taker fees and
  mark-to-market. Each fill is split into:
  - spread capture: `side × (mid at fill − fill price) × qty`;
  - adverse selection at horizon h: `side × (mid at t+h − mid at fill) × qty`;
  - inventory PnL (the residual).
  The components sum exactly to total PnL.
- `agents.py`:
  - `AvellanedaStoikovAgent`, with `sigma` and `kappa` *estimated from the
    simulator in ticks*: `kappa` comes from a regression of log fill
    intensity on quote distance.
  - An inventory-capped variant.
  - A fixed-spread symmetric quoter.
  - Several makers competing in one book.
- `evaluation.py`: N seeds × M agents on common random numbers, with bootstrap
  confidence intervals on paired differences.

**Tests written first**
- Ledger identity: the PnL components sum to mark-to-market PnL on random
  fill sequences.
- Hand-computed attribution for a two-fill example.
- AS quotes in ticks match the closed form; the capped agent never breaches
  its cap.
- The `kappa` estimator recovers a known exponential fill curve.
- Paired evaluation of an agent against itself gives a zero difference.

**Acceptance.** Every baseline trades. This is the legacy failure: report
fill counts. A results table (PnL, Sharpe, fills, adverse selection, inventory
excursions) with 95% CIs over at least 30 seeds.

## Phase 6: Reinforcement-learning market maker (milestone M5) ✅

**Goal.** A learned quoting policy, compared fairly against Phase 5.

**Deliverables**
- `env.py`: a Gymnasium-style environment (`reset(seed)`, `step(action)`)
  over the simulator.
  - Observation: inventory, time-to-go, spread, microprice offset, imbalance,
    recent flow intensities, and the agent's queue positions.
  - Discrete action grid of (bid offset, ask offset) in ticks.
  - Reward: ΔMTM − λ·inventory² (inventory-dampened, Spooner et al. 2018).
- `rl.py` (optional `rl` extra, PyTorch): Double DQN with a replay buffer and
  target network. Train, validation and test use disjoint seed ranges.
  Checkpoints are written to the run directory.
- Evaluation against Phase 5 baselines on the test seeds, with PnL
  attribution.

**Tests written first**
- The environment follows the API contract: shapes, determinism under a seed,
  and termination at the horizon.
- The reward equals the ledger's MTM change minus the penalty.
- The DQN learns a trivial bandit environment. This is a smoke test that
  learning works, run in CI only when torch is installed.

**Acceptance.** Training is reproducible from a config and seed. The test-seed
comparison table is reported with CIs and attribution, *whichever way it comes
out*.

## Phase 7: Productionization ✅

**Deliverables**
- A `microstructure` console script with subcommands `replay`, `calibrate`,
  `simulate`, `evaluate` and `train`, all driven by TOML configs in
  `configs/`.
- Run directories (`runs/<timestamp>-<name>/`) containing a manifest
  (resolved config, git SHA, seeds, package versions), logs and results JSON.
- Structured logging through `logging`, quiet by default.
- Property-based tests (Hypothesis) for book invariants under random flow.
- Coverage gate in CI.
- `pre-commit` configuration.
- Benchmark script with recorded events per second for replay and simulation.

**Acceptance.** A fresh clone can reproduce every number in RESULTS.md from
documented commands, given the data files.

**Added during implementation.** `tests/test_architecture.py` enforces the
layer diagram by parsing imports, and keeps the core free of I/O.

## Beyond the plan

Extensions deliberately left out of scope:

- Queue-reactive intensities (Huang, Lehalle & Rosenbaum 2015).
- Intraday seasonality in `mu`.
- Power-law kernels.
- Multi-asset flow.
- A C or Rust book engine (see `lob-engine-c`) behind the same interface.

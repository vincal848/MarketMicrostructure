# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/). Each roadmap phase
(see [docs/ROADMAP.md](docs/ROADMAP.md)) is logged in two steps: the tests
that define it, then the implementation that makes them pass.

## [Unreleased]: Phase 3b, multi-timescale kernels

Added after Phase 3's first real-data calibration. With one exponential
kernel, the fitted decay hit its upper bound (5000/s) in every window of
both datasets. On 2019-01-30 SPY, 8% of consecutive session events share a
nanosecond timestamp (mostly the cancel and add halves of replaces), 61% of
gaps are under 100 µs, and 98% are under 100 ms. One exponential cannot
span microseconds to seconds.

### Tests (written first, failing)
- `test_hawkes_multiscale.py`:
  - Params hold U components as (U, K, K), and single kernels are promoted
    to U = 1; the branching matrix sums the components; mismatched
    component shapes are rejected.
  - The multi-component likelihood equals the naive oracle.
  - Two-scale simulation rates match the stationary intensity.
  - A fit on a fixed decay grid recovers both scales within 4 SE.
  - Two scales beat either single scale on AIC.
  - Residuals under the true two-scale model pass KS.
  - Online excitation adds every component's jump.

### Implementation
- `hawkes.py`: `HawkesParams` stores `alpha`/`beta` as (U, K, K), promoting
  (K, K) input, and adds `n_components`; the branching matrix sums the
  components. Kernel sums are (n_i, U, K), compensator tails (U, K, K), and
  the likelihood uses einsum. `simulate` is now a loop over `OnlineHawkes`
  rather than a second copy of the thinning algorithm.
- `hawkes_estimation.py`: `fit(stream, K, decay=[d_1, ..., d_U])` fits one
  component per decay (concave, n_params = K + U·K²); residuals sum over
  components; `fit_decay` is unchanged for a single profiled decay.
- `calibration.py`: kernel specs `ProfiledDecay(bounds)` and
  `DecayGrid(decays)`; summaries report `decays`.
- `cli.py`: `--decays` (default 1e5, 1e4, 1e3, 100, 10, 1 per second) and
  `--single-decay` for the old profiled fit.
- Test changes made during implementation:
  - Updated to (U, K, K) indexing: the naive oracle in
    `test_hawkes_estimation.py`, the excitation assertion in
    `test_hawkes.py`, and `beta[0, 0, 0]` in the decay assertions.
  - `calibrate_window` now takes a kernel spec instead of `decay_bounds`;
    the summary key `decay` became `decays`; added a decay-grid
    calibration test.
  - The two-scale test process was made less critical (spectral radius
    0.86 → 0.72). With seed 21 the old process ran 11.5% below its
    stationary rate. Over 12 seeds the mean rate was 2.877 against 2.889
    theoretical, so the simulator is unbiased and the tolerance was the
    problem: about 2.6σ at the old criticality, about 4σ at the new one.

## Phase 4, generative simulator

### Tests (written first, failing)
- `test_hawkes.py`: `OnlineHawkes`. Thinning event rates match the
  stationary intensity; `excite(j)` raises the intensity by exactly
  `alpha[:, j]`; `next_event(until)` stops at the requested time.
- `test_simulator.py`:
  - Generated background rates are within 5% of the stationary intensity.
  - The book is never crossed; the same seed reproduces the run and a
    different seed does not.
  - Applied flow uses only the calibrated mark values.
  - Background cancels never touch agent orders.
  - Every agent order reaches the book exactly `latency` after it was sent.
  - Agent fills carry side, price and size, and passive fills are flagged
    non-aggressive.
  - Unchanged quotes keep their order ids, and so their queue position.
    Revised before implementation: the first version depended on how
    often the touch moves, which made it flaky by design. It now uses a
    quoter re-sending identical far quotes, which must create exactly two
    orders.
  - An agent market order excites the background MB intensity, and its
    fills are aggressive.
- `test_stylized.py`:
  - The spread distribution buckets ticks and caps the tail.
  - Alternating trade signs give lag-1 autocorrelation −1.
  - The signature plot is checked on a hand-computed path.
  - Total-variation distance.
  - `record_tape` from replayed events: quotes recorded only when both
    sides exist; trades signed by aggressor or by mid for hidden trades.

### Implementation
- `hawkes.OnlineHawkes`: event-at-a-time thinning with `advance_to`,
  `excite` and `next_event(until)`. Valid because thinning proposals are
  memoryless at a deadline.
- `simulator.py`: `MarketSimulator`.
  - Background Hawkes events map to book operations with sampled marks.
  - Background cancels select only background orders.
  - Agents follow an `Agent` protocol (`decide`, `on_fill`,
    `decision_interval`) and get a `MarketView` that includes their own
    queue positions.
  - Actions are `Quote` (identical orders kept, so queue priority holds)
    and `SendMarketOrder` (excites the flow). Order arrival is delayed by
    the latency; agents are woken on their own fills.
  - The result is a frozen `SimulationResult` with the tape, the applied
    flow, generated and skipped counts, and the order log.
- `stylized.py`: `MarketTape`, `record_tape` (real data), spread
  distribution, trade-sign ACF, signature plot, total variation, and a
  `summarize` JSON summary.
- Test changes made during implementation:
  - `test_background_cancels_never_touch_agent_orders` compares sets:
    `Side` is a `StrEnum`, so sorting put "ask" before "bid" and the
    expected list order was wrong.
- Design fix: the simulator first built a placeholder `SimulationResult`
  at construction. It now keeps counters and builds the immutable result
  once, in `_finish`.

## Phase 3, event classification and calibration

### Tests (written first, failing)
- `test_flow.py`:
  - A sweep across levels at one timestamp is a single MB; executions at
    different times stay separate; executing a resting bid is an MS.
  - Adds are split into LA/LI/LD by their position relative to the best,
    with tick distances; the first order on an empty side is LA.
  - Cancels record their distance from the best.
  - A replace is C followed by an add, classified against the book after
    the cancel.
  - Hidden trades are signed by price against mid, and at-mid trades are
    counted as unsigned; cross executions are excluded, printable ones
    included.
  - Unknown ids are skipped and reported.
  - `between` uses half-open windows; `to_stream` produces seconds from the
    window start; `marks` collects sizes and distances.
- `test_calibration.py`:
  - `session_windows` tiles the session.
  - A known six-type process is recovered end to end (decay within 4 SE,
    AIC better than Poisson, KS per type).
  - Empty types are tolerated.
  - Summaries are JSON-serializable.
- `test_databento.py`, with a generated MBP-10 CSV reproducing the real
  quirks:
  - side-`N` adds; a replace published only as its add;
  - fills whose level decrease arrives on a later record, or on a trade
    record; merged same-time trades; an unsigned trade signed by mid;
  - the 11th level sliding into the top 10 is not an add;
  - New York local timestamps; chunked reading does not split events.

### Implementation
- `flow.py`:
  - `FlowType` (MB, MS, LA, LI, LD, C) and `ClassifiedFlow` (parallel int
    arrays with `between` and `to_stream`); `FlowMarks` with `marks`.
  - `classify` replays events and types each one against the book before
    it, merging same-time same-aggressor executions.
  - `FlowRecorder` and `add_type` are shared with the Databento adapter.
- `databento.py`: `classify_mbp10`/`read_mbp10` classify by snapshot diffs.
  - Trades register pending fills that later level decreases absorb.
  - Only prices visible in both snapshots are compared.
  - Records flagged as snapshots, or `R` clears, re-seed the book without
    emitting flow.
  - Timestamps are converted to New York time.
- `calibration.py`: `session_windows`; `calibrate_window`, which fits Hawkes
  (profiled decay) and Poisson and returns a `WindowFit` with AIC gain,
  branching ratio, per-type KS, and a JSON `summary`.
- `cli.py`: `calibrate-itch` and `calibrate-mbp10` (per-window JSON plus an
  optional `.npz` of session mark samples), and `logging` progress output
  (`-q` to silence).
- Test changes made during implementation:
  - Corrected `test_marks_collect_sizes_and_distances_per_type`: the fixture
    book's order 2 is itself an LD add, so the expected LD marks are
    `[10, 10, 9]`, not `[10, 9]`.
  - The Databento slide test asserts on the second row's events only,
    since the first row legitimately adds ten bid levels.
  - The MBP fixture builder moved to `tests/mbp_writer.py` for reuse.
  - Added CLI tests for both calibration commands.

## Phase 2, Hawkes estimation

### Tests (written first, failing)
- `test_hawkes_estimation.py`:
  - `log_likelihood` must equal an O(n²) oracle written directly from the
    definition, for a 2-D process with four different decays.
  - Parameter recovery within 4 standard errors, in 1-D and 2-D with the
    decay fixed, and for a profiled shared decay (including the decay's
    own SE).
  - The fitted log-likelihood is at least the truth's, and is reproduced by
    `log_likelihood` at the fitted params.
  - Time-rescaled residuals pass KS under the true model; residuals equal
    compensator increments in a hand-computed case.
  - On Hawkes data, Poisson loses on AIC and fails KS.
  - On Poisson data, the Hawkes fit finds a spectral radius below 0.1.
  - Poisson MLE equals event rates, with √n/T standard errors.
  - Non-positive decays and out-of-range types are rejected.

### Implementation
- `hawkes.py`: vectorized kernel sums (`target_sums`, `compensator_tails`).
  The recursion is evaluated with blockwise cumulative sums, rebased so
  `exp(beta * dt)` never overflows. `log_likelihood` now uses them instead
  of a per-event Python loop; the O(n²) oracle test pins its correctness.
- `hawkes_estimation.py` (new module, keeping estimation out of the model
  module):
  - `fit` with a fixed shared decay: the concave per-target problems are
    solved by L-BFGS-B with the exact gradient, and standard errors come
    from the observed information.
  - `fit_decay`: profile likelihood over the decay, with an SE from the
    profile's curvature.
  - `fit_poisson`, `HawkesFit.aic`, `rescaled_residuals`, `ks_exponential`.
- Dependencies: `scipy` (runtime), `scipy-stubs` (dev).
- Test changes made during implementation:
  - The estimation tests import from `hawkes_estimation` rather than
    `hawkes`.
  - Added `test_objective_gradient_matches_finite_differences` (written and
    seen failing before `_objective` was factored out). The roadmap listed
    it, but the first test commit missed it.

## Phase 1, order-level data and exact replay

### Acceptance on real data (M1 met)
- Full-day replay of SPY from the Nasdaq ITCH sample day 2019-01-30:
  3,065,003 events, 130,532 audited executions, zero priority violations,
  zero unknown ids, zero quantity mismatches, zero crossing adds, 230 s
  wall time. Details in `docs/RESULTS.md`.

### Tests (written first, failing)
- `test_book.py`: `execute_order` fills a named order anywhere in its queue
  and keeps queue priority on partial fills; over-execution is rejected
  without changing state; `queue_ahead` counts only older orders at the
  same price; `would_cross` agrees with the matching rule; `resting_order`
  returns a detached copy.
- `test_events.py`: events are immutable values; non-positive quantities
  and negative timestamps are rejected.
- `test_itch.py` with `itch_writer.py` (a byte-level ITCH 5.0 builder):
  decoding of S/R/A/F/E/C/X/D/U/P; ignored types are skipped by length; the
  symbol filter; gzip and plain files; full 6-byte timestamps; an unknown
  symbol is an error; truncation is an error; messages straddling read
  chunks.
- `test_replay.py`: depth after a hand-built stream; a priority audit that
  flags executions behind the queue head or away from the best price;
  unknown-id references counted, not raised; replace = new order at the back
  of the queue; crossing adds counted; hidden trades leave the book alone;
  `C` executions applied but not audited; audit restricted to the 'Q'–'M'
  session; `seed_book`; exact replay of both LOBSTER fixtures against every
  snapshot, including a window that opens with resting orders; mismatches
  reported by row.
- `test_lobster.py`: `to_events` mapping of types 1–4; pre-window order ids
  mapped onto per-level seed orders; `depths` drops empty-level codes.
- New fixture `seeded_message.csv`/`seeded_orderbook.csv`: a 2-level window
  whose first snapshot already holds orders added before the window.

### Implementation
- `events.py`: frozen, validated event dataclasses; the `OrderEvent` union;
  `seed_order_id`.
- `book.py`: `execute_order`, `queue_ahead`, `is_at_front_of_best`,
  `would_cross`, `resting_order`, `__contains__`; `Depth` alias.
- `itch.py`: chunked streaming decoder for ITCH 5.0. Other symbols'
  messages are skipped after reading two bytes; gzip is detected from magic
  bytes.
- `replay.py`: `Replayer`/`replay` with a `ReplayReport` audit (priority
  violations, unknown ids, quantity mismatches, crossing adds, hidden
  trades, executions with price); `seed_book`; `verify_snapshots`.
- `lobster.py`: `to_events` (redirects pre-window ids onto seed orders; maps
  type 6/7 to `SystemEvent("cross"/"halt")` so rows stay aligned) and
  `depths`.
- `cli.py`: `microstructure replay-itch` writes a JSON audit report and
  exits 1 when the audit is not clean. It is registered as a console script.
- Test changes made during implementation:
  - `verify_snapshots` takes `n_levels` explicitly. Inferring the depth
    from snapshots would break when no row has every level filled.
  - Added an oversized-cancel quantity-mismatch test, `test_cli.py`, and
    `test_real_data.py`. The last is marked `data`: skipped unless the ITCH
    file exists locally, and deselected by default.

## [0.2.0] - 2026-10-07: Phase 0, foundation

Restructured from flat scripts to a typed package after an architecture and
code-quality review. Process note: in this phase the module rewrites were
written before their tests. From Phase 1 on, every phase lands tests first.

### Review findings fixed (each with a regression test)
- `lobster.read_orderbook`: the column-count check could never fire. pandas,
  given fewer names than columns, moves the surplus leading columns into the
  index, so a 2-level file read as 1 level silently returned level 2
  labelled as level 1. Column counts are now checked on the raw frame.
- `OrderBook.market_order`: any side other than `"bid"` (for example
  `"buy"`) was treated as a sell. `add_limit_order` with a mistyped side
  could trade against the book before raising. Sides are now a `Side` enum,
  validated before any state changes.
- `hawkes`: negative `alpha` was accepted, which allows negative
  intensities. Mismatched `mu`/`alpha`/`beta` shapes surfaced only as numpy
  broadcast errors. `beta = 0` divided by zero. `log_likelihood` returned a
  finite value for events after the horizon. All are now rejected on
  construction of `HawkesParams` / `EventStream`.
- `OrderBook.best_bid` / `best_ask` were `max`/`min` over every level. Each
  matching step called them, so a sweep through L levels cost O(L²). Levels
  are now kept sorted, and the best price is O(1).

### Changed
- Flat modules moved to `src/microstructure/`: `lob.py` → `book.py`,
  `agents.py` → `avellaneda_stoikov.py`; `hawkes.py` and `lobster.py` kept
  their names. History is preserved with `git mv`.
- `book`: `Order`/`Fill` are dataclasses. A `_BookSide` class with a single
  rank function replaces duplicated per-side branches, and limit and market
  orders share one `_match` loop. Integer prices are enforced with
  `operator.index`.
- `hawkes`: `HawkesParams` and `EventStream` are validated, immutable value
  objects. `simulate` returns an `EventStream`. `log_likelihood(stream,
  params)` replaces six loose arguments. Seeded simulations reproduce the
  previous version's output exactly.
- `avellaneda_stoikov`: `ASParams` is validated once, `time_to_go` replaces
  the `(T, t)` pairs, and `quotes` returns a `Quote`. The units contract is
  documented and enforced by a price-rescaling invariance test.
- `lobster`: `EventType` and `Direction` are `IntEnum`s. Unknown directions
  are rejected.
- Tests import the installed package. The `sys.path` edits are gone.
- `docs/DESIGN.md` → `docs/ARCHITECTURE.md`, extended with the layer rules
  and design decisions.

### Added
- `pyproject.toml` (hatchling) with a `dev` extra; `py.typed`.
- CI `lint` job: `ruff check`, `ruff format --check`, `mypy --strict`.
- Tests: a differential test of the book against a brute-force depth
  reference under 2,000 random operations; a closed-form Poisson likelihood
  check; seed reproducibility; parameter immutability.
- `docs/MOTIVATION.md` and `docs/ROADMAP.md`.

### Removed
- `fit_mle` and `AvellanedaStoikovAgent` stubs, which only raised
  `NotImplementedError`, along with the tests asserting that. The roadmap
  now records the planned work instead.
- `requirements.txt` / `requirements-dev.txt`, replaced by `pyproject.toml`.

## [0.1.1] - 2026-10-07
### Added
- `legacy/cmu_ml2_meta_dqn/`: the CMU ML2 Meta-DQN prototype, with results
  and the caveats found when it was archived.

## [0.1.0] - 2026-10-01
### Added
- Scaffold: order book, Hawkes thinning and likelihood, Avellaneda-Stoikov
  formulas, LOBSTER parser, tests and CI.

# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/). Each roadmap phase
(see [docs/ROADMAP.md](docs/ROADMAP.md)) is logged in two steps: the tests
that define it, then the implementation that makes them pass.

## [Unreleased]: Phase 2, Hawkes estimation

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

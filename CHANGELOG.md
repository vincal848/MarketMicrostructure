# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/). Each roadmap phase
(see [docs/ROADMAP.md](docs/ROADMAP.md)) is logged in two steps: the tests
that define it, then the implementation that makes them pass.

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

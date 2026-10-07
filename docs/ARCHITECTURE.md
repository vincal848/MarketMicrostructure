# Architecture

## Layers

The package is organised as layers with a single dependency direction. A
module may import from its own layer or lower ones, never higher.
`tests/test_architecture.py` parses every module's imports and enforces
this, so the diagram cannot drift from the code:

```
  7  experiment · bench · cli                  entry points, run directories
  6  evaluation · env · rl                     experiments over the simulator
  5  agents · accounting                       market makers, PnL ledger
  4  simulator                                 discrete-event market
  3  flow · calibration · databento · stylized real flow -> event alphabet -> fit; tape statistics
  2  replay                                    drive a book from order events
  1  itch · lobster                            order-level source adapters
  0  events · book · hawkes · hawkes_estimation · avellaneda_stoikov
                                               core: pure, typed, no I/O
```

`databento` sits in layer 3, not with the other adapters. MBP-10 data has no
order ids, so it cannot be replayed; it is classified straight into flow by
snapshot diffs.

Rules that keep the boundaries clean:

- **Core modules do no I/O.** Layer 0 never opens files, prints, logs, or
  imports `os`, `sys`, `pathlib` or `subprocess`; the architecture test
  checks this. Everything that touches the filesystem lives in an adapter
  or an entry point, which is what lets the core be tested exhaustively and
  quickly.
- **One event model.** Every order-level source is converted to `events.py`
  types at the boundary. Nothing above the adapters knows whether data came
  from ITCH or LOBSTER.
- **Integer prices everywhere below the agents.** Prices are integer 1/10000
  dollars (ITCH and LOBSTER units), and agents work in integer ticks.
  Floats appear only in model parameters and PnL. The legacy project's unit
  bug came from mixing these.
- **Validated value objects.** Parameters (`HawkesParams`, `ASParams`,
  `SimulationConfig`, `EnvConfig`, the experiment specs) and data
  (`EventStream`, `ClassifiedFlow`) validate once, on construction, and are
  immutable. Functions take these objects instead of loose arrays, so every
  function can rely on its inputs.
- **Optional heavy dependencies stay optional.** PyTorch is imported only by
  `rl.py`, and by `experiment.py` only when a config contains `[rl]`.

## Modules

| Module | Layer | Contents | Key invariant or contract |
|---|---|---|---|
| `events.py` | 0 | Normalized order events, `seed_order_id` | Immutable; quantities positive, timestamps non-negative |
| `book.py` | 0 | `Side`, `Order`, `Fill`, `OrderBook` | best bid < best ask after every call; a rejected call changes nothing |
| `hawkes.py` | 0 | `HawkesParams` (U components), `EventStream`, `OnlineHawkes`, `simulate`, `log_likelihood`, vectorized kernel sums | Params non-negative and read-only; stepping `OnlineHawkes` reproduces one run exactly |
| `hawkes_estimation.py` | 0 | `fit` (decay grid), `fit_decay` (profiled), `fit_poisson`, residuals, KS | Concave per-target MLE with exact gradient; SEs from observed information |
| `avellaneda_stoikov.py` | 0 | `ASParams`, `reservation_price`, `optimal_spread`, `quotes` | Outputs rescale exactly with the price unit |
| `itch.py` | 1 | Streaming TotalView-ITCH 5.0 decoder | Truncated streams and unknown symbols are errors |
| `lobster.py` | 1 | LOBSTER parser, `to_events`, `depths` | Column counts checked on the raw file; one event per row |
| `replay.py` | 2 | `Replayer` with a price-time priority audit, `seed_book`, `verify_snapshots`, `depth_at` | Every execution is checked against the queue head at the best price |
| `flow.py` | 3 | `FlowType`, `ClassifiedFlow`, `classify`, `marks` | Each event is typed against the book *before* it |
| `databento.py` | 3 | MBP-10 snapshot-diff classifier | Fills absorbed before cancels; only prices visible in both snapshots compared |
| `calibration.py` | 3 | `session_windows`, `calibrate_window` (`ProfiledDecay` / `DecayGrid`) | Hawkes and Poisson fitted to the same window |
| `stylized.py` | 3 | `MarketTape`, `record_tape`, spread / ACF / signature plot | Same code for real and simulated tapes |
| `simulator.py` | 4 | `MarketSimulator`, `Agent` protocol, `Quote`, `SendMarketOrder` | Background cancels never touch agent orders; latency exact; seeded |
| `accounting.py` | 5 | `Ledger`, `MidPath`, PnL attribution | Spread + adverse + inventory - fees = MTM exactly |
| `agents.py` | 5 | `AvellanedaStoikovAgent`, `FixedSpreadAgent`, `estimate_sigma`, `estimate_fill_curve` | Ticks throughout; post-only; cap sizes orders down |
| `evaluation.py` | 6 | `Scenario`, `evaluate`, `EvaluationTable`, `calibrate_avellaneda_stoikov`, bootstrap CIs | Paired by seed; every agent rebuilt per run |
| `env.py` | 6 | `MarketMakingEnv` | Same accounting as the baselines; inventory cap at every decision |
| `rl.py` | 6 | Double DQN, `ReplayBuffer`, `Policy` | Seeded and bit-for-bit reproducible |
| `experiment.py` | 7 | TOML config, `build_scenario`, `run_experiment` | Disjoint seed ranges; manifest pins code and versions |
| `bench.py` | 7 | Throughput benchmarks | Seeded synthetic workloads |
| `cli.py` | 7 | `microstructure` console script | Thin wrappers over the layers below |

## Order book

`OrderBook` keeps one `_BookSide` per side. Each side stores
`price -> deque[Order]` plus a list of prices sorted worst-to-best:

- Best price is `_prices[-1]`: O(1).
- Consuming the top level is `pop()`: O(1).
- Adding a new level is `insort`: O(L) memmove. That is acceptable for the
  tens to low hundreds of live levels in an equity book.
- Within a level, FIFO is a `deque`. A full cancel is O(queue length), and
  queues are short in practice.

"Better" means a higher price for bids and a lower price for asks.
`_BookSide._rank` maps both onto "larger is better", so matching, crossing
checks and depth snapshots are each written once. Earlier versions repeated
`if side == "bid"` branches throughout, and an unknown side string was
silently treated as `"ask"`.

## Event alphabet (Hawkes dimensions)

Order flow is modelled as a K = 6 point process. Every source maps onto these
types. The book state *before* the event decides between at / inside / away.

| Type | Meaning | Book effect | ITCH 5.0 | LOBSTER | Hawkes dim |
|---|---|---|---|---|---|
| MB | Market buy | consume asks | `E`/`C` executing a sell order (merged per timestamp); `P` buy | 4/5, direction −1 | 0 |
| MS | Market sell | consume bids | `E`/`C` executing a buy order (merged per timestamp); `P` sell | 4/5, direction +1 | 1 |
| LA | Limit add at the best | join the touch | `A`/`F` at best | 1 at best | 2 |
| LI | Limit add inside the spread | improve the touch | `A`/`F` strictly inside | 1 strictly inside | 3 |
| LD | Limit add away from the best | deeper queue | `A`/`F` behind best | 1 behind best | 4 |
| C | Cancel (full or partial) | remove resting volume | `X`, `D`; `U` = C then add | 2, 3 | 5 |

Hidden executions (ITCH `P`, LOBSTER 5) count as market orders for the point
process but have no effect on the displayed book.

## Hawkes process

For event type `i`, given the history of all `K` types:

```
lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} alpha_ij * exp(-beta_ij * (t - t_k^j))
```

- `mu_i` is the exogenous background rate of type `i`.
- `alpha_ij` is the jump in type-`i` intensity caused by one type-`j` event.
- `beta_ij` is the decay rate of that jump.

The branching matrix `n = alpha / beta` gives the expected number of direct
children. The process is stationary iff `spectral_radius(n) < 1` (Hawkes &
Oakes 1974). In that case the long-run intensity is `(I - n)^-1 mu`, which
the tests use to check long simulations.

Both `simulate` and `log_likelihood` carry an excitation matrix `R[i, j]` (the
decayed contribution of type `j` to type `i`). Exponential kernels make `R`
Markov: advancing it in time is one elementwise multiply. That keeps both
algorithms at O(n_events · K²).

## Design decisions

| Decision | Alternative rejected | Why |
|---|---|---|
| `src/` layout, installed package | Flat modules plus `sys.path` edits in tests | Tests exercise the installed package exactly as users import it |
| Frozen, validated parameter objects | Loose arrays re-coerced in every function | One validation point; contradictory shapes or negative excitation fail loudly at construction |
| Hawkes `fit` with a fixed decay and profiled `beta` (Phase 2) | Joint non-convex optimisation over all of `(mu, alpha, beta)` | Concave inner problem with a unique optimum (the approach of the `tick` library) |
| MBP-10 treated as snapshots + trades | Replaying MBP actions as if they were order-level | MBP omits the cancel half of a replace, so action replay cannot be exact |
| No framework for RL beyond PyTorch | stable-baselines3, RLlib | The policy and environment are small, and owning the training loop keeps evaluation auditable |

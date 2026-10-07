# Architecture

## Layers

The package is organised as layers with a single dependency direction. A
module may import from layers below it, never above:

```
  cli / configs                         Phase 7   entry points, run manifests
        │
  evaluation · rl · env                 Phases 5–6  experiments over the simulator
        │
  agents · accounting                   Phase 5   market makers, PnL ledger
        │
  simulator · stylized                  Phase 4   discrete-event market
        │
  calibration · flow                    Phase 3   real flow → event alphabet → fit
        │
  replay                                Phase 1   drive a book from order events
        │
  itch · lobster · databento            Phases 1/3  source adapters (I/O lives here)
        │
  events · book · hawkes · avellaneda_stoikov     core: pure, typed, no I/O
```

Rules that keep the boundaries clean:

- **Core modules do no I/O.** `book`, `hawkes`, `avellaneda_stoikov` and
  `events` never read files, log or print. Everything that touches the
  filesystem lives in an adapter or the CLI. This is what lets the core be
  tested exhaustively and quickly.
- **One event model.** Every data source is converted to `events.py` types at
  the boundary. Nothing above the adapters knows whether data came from ITCH,
  LOBSTER or Databento.
- **Integer prices everywhere below the agents.** Prices are integer ticks
  (or 1/10000 dollars at the adapter boundary). Floats appear only in model
  parameters and PnL. The legacy project's unit bug came from mixing these.
- **Validated value objects.** Parameters (`HawkesParams`, `ASParams`) and
  data (`EventStream`) validate once, on construction, and are immutable.
  Functions take these objects instead of loose arrays, so every function
  can rely on its inputs.

## Current modules

| Module | Contents | Key invariant |
|---|---|---|
| `book.py` | `Side`, `Order`, `Fill`, `OrderBook` | best bid < best ask after every call; a rejected call changes nothing |
| `hawkes.py` | `HawkesParams`, `EventStream`, `simulate`, `log_likelihood` | params are non-negative, shapes consistent, read-only; events sorted and inside `[0, horizon]` |
| `avellaneda_stoikov.py` | `ASParams`, `reservation_price`, `optimal_spread`, `quotes` | outputs rescale exactly with the price unit |
| `lobster.py` | `EventType`, `Direction`, `read_messages`, `read_orderbook`, `read_paired` | column count checked against the raw file |

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

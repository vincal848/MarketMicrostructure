# Design notes

## Event types

The simulator treats order flow as a multivariate point process over a
small alphabet of event types, each of which maps to one dimension of the
Hawkes process in `hawkes.py` and to one operation on the book in `lob.py`.
LOBSTER's own message types (see `lobster.py`) collapse onto this alphabet;
the mapping below is what M3 (calibration) will use to bucket real events.

| Event type | Book effect | LOBSTER type(s) | Hawkes dimension |
|---|---|---|---|
| Market buy | `market_order("bid", qty)` | type 4/5, direction -1 | MB |
| Market sell | `market_order("ask", qty)` | type 4/5, direction 1 | MS |
| Limit add at the best | `add_limit_order` at `best_bid`/`best_ask` | type 1, price = top of book | LA |
| Limit add inside the spread | `add_limit_order` strictly between best bid and ask | type 1, price between quotes | LI |
| Limit add away from the best | `add_limit_order` worse than the current best | type 1, price beyond top of book | LD |
| Cancel | `cancel_order` (full or partial) | type 2 (partial) or 3 (full) | C |

So K = 6 for the full model. `hawkes.py` itself is dimension-agnostic (K is
just `len(mu)`); this table is the convention the calibration step (M3) will
use to turn a LOBSTER message file into six timestamp sequences.

## Hawkes intensity

For event type `i`, given the full history of all `K` types:

```
lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} alpha_ij * exp(-beta_ij * (t - t_k^j))
```

- `mu_i` -- exogenous background rate of type `i`.
- `alpha_ij` -- how much one type-`j` event raises type-`i`'s intensity, on
  arrival.
- `beta_ij` -- how fast that boost decays.

Economically: a market sell (MS) should strongly self-excite (orders arrive
in bursts) and excite cancels on the bid side (market makers pulling quotes
after adverse flow) and limit adds on the ask side (replenishment) -- all of
that is encoded in the off-diagonal structure of `alpha` and `beta`, not
hand-wired into the simulator. M2 checks that MLE can recover a known
`(mu, alpha, beta)` from simulated data; M3 fits it to the LOBSTER sample.

The branching matrix `n_ij = alpha_ij / beta_ij` is the expected number of
direct type-`i` children of one type-`j` event. The process is stationary
iff `spectral_radius(n) < 1` (Hawkes & Oakes, 1974); `hawkes.simulate`
enforces this before running. The stationary intensity is then the vector
`(I - n)^-1 mu`, which is also how `test_hawkes.py` checks a long simulation
against its theoretical event rate.

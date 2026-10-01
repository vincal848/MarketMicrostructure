# Market Microstructure

[![tests](https://github.com/vincal848/MarketMicrostructure/actions/workflows/tests.yml/badge.svg)](https://github.com/vincal848/MarketMicrostructure/actions/workflows/tests.yml)

**Hawkes-driven limit order book simulator with competing market makers.**

I want a discrete-event, tick-driven limit order book simulation built on
real LOBSTER (NASDAQ ITCH-derived) sample data, with several market makers
quoting into it at once. The original framing of this repository was just
"simulate a LOB with market makers"; what makes it a specific project rather
than a vague one is modelling order flow itself as a multivariate Hawkes
process -- fit by maximum likelihood to LOBSTER data rather than assumed --
and eventually replacing the textbook market maker with a reinforcement
learning agent that quotes against that simulated flow. Latency, inventory
limits, and adverse selection all fall out of that setup rather than being
bolted on separately.

This is a scaffold, not a working simulator yet. The point right now is to
get the pieces that the first tests need right -- the book engine, the
Hawkes machinery, the Avellaneda-Stoikov formulas, the LOBSTER parser -- and
to write down what "done" means for each later milestone before I build it.

## Motivation

A limit order book is a queueing system: orders arrive, rest, get matched
or cancelled, in a strict price-time priority. Most LOB simulators either
replay historical data verbatim (so you can't ask "what if a market maker
had been quoting here") or generate order flow from an unconditional Poisson
process (so bursts of market orders and clustered cancellations -- the stuff
that actually drives adverse selection -- don't show up). A multivariate
Hawkes process is the natural middle ground: each event type (market buy,
market sell, limit adds at/inside/away from the best, cancels) has a
baseline rate plus self- and cross-excitation from recent events of every
type, and it can be calibrated to a real order flow by maximum likelihood
rather than hand-tuned. That gives a simulator that is generative (I can run
it forward under a market maker's influence) but still anchored to data
(the calibration step has to match LOBSTER's own statistics before I trust
anything built on top of it).

## Method

**Book.** Price-time priority, integer tick prices, FIFO within a level.
Four operations: add a limit order (crossing orders match immediately, the
remainder rests), cancel (full or partial), a market order that walks
levels until filled or the book runs dry, and a depth snapshot to N levels.
This is `lob.py`.

**Order flow.** A multivariate Hawkes process over event types (market
buy/sell, limit add at/inside/away from the best, cancel -- see
`docs/DESIGN.md` for the full table). Exponential kernels, so the intensity
of type `i` is

```
lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} alpha_ij * exp(-beta_ij (t - t_k^j))
```

Simulated by Ogata's thinning algorithm; calibrated to LOBSTER by maximum
likelihood. This is `hawkes.py`.

**Market makers.** Avellaneda-Stoikov as the baseline: closed-form
reservation price and optimal spread under exponential utility and Poisson
fill intensity. An inventory-limited variant on top of that (refuse to quote
further on the side that would breach a position cap). Latency modelled as
a fixed or random delay between an agent deciding to quote and that quote
actually reaching the book, which matters a lot once several market makers
are racing each other. Eventually a reinforcement-learning agent that
quotes directly against the simulated Hawkes flow instead of using the
closed-form formulas. This is `agents.py`.

**Data.** LOBSTER message and orderbook files, parsed into pandas
DataFrames and checked for row alignment. This is `lobster.py`.

## Data

[LOBSTER](https://lobsterdata.com) provides free sample files (one trading
day, a handful of large-cap tickers) reconstructed from NASDAQ ITCH, split
into two CSVs per day with no header, aligned row for row:

- **Message file** (6 columns): `time, type, order_id, size, price,
  direction`. `type` is 1 (new limit order), 2 (partial cancellation), 3
  (full deletion), 4 (visible execution), 5 (hidden execution), 6 (cross
  trade), or 7 (trading halt). `price` is dollars x 10000, as an integer.
  `direction` is 1 (buy) or -1 (sell) and refers to the resting limit
  order's side, so an aggressive market buy appears as an execution of a
  sell limit order.
- **Orderbook file** (4 x N columns, N = requested depth): `ask_price_i,
  ask_size_i, bid_price_i, bid_size_i` for level `i = 1..N`, best first.

Full column specs are documented in `lobster.py`'s module docstring. Sample
files live under `data/`, which is gitignored -- LOBSTER's free tier is
redistributable for personal use only, so they are downloaded locally
rather than committed. `tests/fixtures/` holds a tiny, hand-made CSV pair in
the same format so the parser tests run without any real data or network
access.

## Milestones

- [ ] **M1** -- Book engine + replay of LOBSTER messages reproduces
      LOBSTER's own orderbook snapshots exactly, message by message.
- [ ] **M2** -- Hawkes simulation + MLE recovers known parameters: simulate
      from a chosen `(mu, alpha, beta)`, fit, and check the fit lands close
      to the truth.
- [ ] **M3** -- Calibrate the Hawkes process to the LOBSTER sample data
      itself (bucketing real events into the type table in
      `docs/DESIGN.md`).
- [ ] **M4** -- Market makers (Avellaneda-Stoikov baseline, inventory-limited
      variant, latency) quoting into the simulated book, with PnL,
      inventory, and adverse-selection metrics tracked per agent.
- [ ] **M5** -- Reinforcement-learning agent trained against the simulated
      Hawkes flow, compared against the M4 baselines.

## Success metrics

- **M1:** exact match against LOBSTER's own orderbook snapshot file --
  not "close", every price and size at every level, after every message.
- **M2:** parameter recovery -- fitted `(mu, alpha, beta)` within the known
  rate of statistical error of the simulated truth, and the log-likelihood
  at the fit at least as high as at the truth.
- **M3:** stylised facts reproduced out of sample -- spread distribution,
  the signature plot (realized variance vs sampling frequency), and
  order-flow autocorrelation, compared against the same statistics computed
  directly on LOBSTER data.
- **M4:** market maker PnL decomposed into spread capture vs adverse
  selection (the standard split: how much of PnL comes from the
  bid-ask spread itself vs from being run over by informed flow), per agent,
  per run.
- **M5:** the RL agent's PnL/inventory/adverse-selection profile against the
  M4 baselines under identical simulated flow.

## Status

Scaffold. M1 in progress.

## Quick start

```bash
pip install -r requirements-dev.txt
pytest tests -q
```

## Repository guide

| Path | Contents |
|---|---|
| `lob.py` | Price-time priority order book: add, cancel, market order, depth snapshot |
| `hawkes.py` | Multivariate Hawkes: thinning simulation, stationary intensity, log-likelihood, MLE stub |
| `agents.py` | Avellaneda-Stoikov reservation price and spread; agent-loop stub |
| `lobster.py` | LOBSTER message/orderbook CSV parser |
| `tests/` | One test file per module, plus `tests/fixtures/` for the LOBSTER parser |
| `docs/DESIGN.md` | Event-type table and the Hawkes intensity formula |

## Notes

- No real LOBSTER data is in this repository or in CI. The parser is tested
  against a tiny fixture built by hand; M1's exact-snapshot-match milestone
  is the point at which real sample data gets downloaded and used locally.
- `agents.py`'s closed-form functions assume the Avellaneda-Stoikov setup
  exactly as published (constant volatility, exponential utility, Poisson
  fill intensity at a fixed `kappa`) -- known simplifications, not something
  this scaffold tries to relax yet.

# Market Microstructure

[![ci](https://github.com/vincal848/MarketMicrostructure/actions/workflows/ci.yml/badge.svg)](https://github.com/vincal848/MarketMicrostructure/actions/workflows/ci.yml)

**A Hawkes-driven limit order book simulator for evaluating market makers,
calibrated to and validated against real Nasdaq order flow.**

Historical data cannot tell you how a market maker would have done. Its
orders would have changed the book, its queue position is unknown, and the
fills it would have received are exactly the ones followed by adverse price
moves. This project builds the standard alternative:

1. Reconstruct real order books message by message.
2. Fit a multivariate Hawkes process to the order flow.
3. Simulate that flow forward, with market makers quoting into the same
   book.

The market makers are closed-form (Avellaneda-Stoikov) and learned
(reinforcement learning), and they are compared on identical simulated flow
with PnL split into spread capture, adverse selection and inventory.

The full argument, with references, is in
**[docs/MOTIVATION.md](docs/MOTIVATION.md)**.

## Status

Phases 0–2 are complete. The remaining phases are planned in
**[docs/ROADMAP.md](docs/ROADMAP.md)**, each with tests written before code
and acceptance criteria measured on real data. Progress is logged in
[CHANGELOG.md](CHANGELOG.md).

| Milestone | What "done" means | State |
|---|---|---|
| M1 replay | ITCH day replayed with zero unknown ids and zero price-time violations | **met**: 130,532 executions audited, 0 violations ([results](docs/RESULTS.md)) |
| M2 estimation | MLE recovers simulated parameters within 4 SE; residuals pass KS | **met** (CI tests) |
| M3 calibration | Six-type Hawkes fit to real flow beats Poisson; stylized facts compared | planned (Phases 3–4) |
| M4 market makers | Calibrated baselines trade; PnL attributed, with CIs over ≥30 seeds | planned (Phase 5) |
| M5 RL agent | Learned policy compared to baselines on held-out seeds | planned (Phase 6) |

## Quick start

```bash
pip install -e ".[dev]"
pytest -q                 # tests
ruff check . && mypy      # lint + strict types
```

```python
from microstructure.book import OrderBook, Side
from microstructure.hawkes import HawkesParams, simulate, log_likelihood
from microstructure.avellaneda_stoikov import ASParams, quotes

book = OrderBook()
book.add_limit_order(1, Side.ASK, price=10_001, qty=100)
fills, leftover = book.market_order(Side.BID, 60)

params = HawkesParams(mu=[0.3, 0.2], alpha=[[0.4, 0.1], [0.2, 0.3]], beta=[[1.0, 1.0], [1.0, 1.0]])
stream = simulate(params, horizon=1_000.0, seed=0)
print(params.spectral_radius(), len(stream), log_likelihood(stream, params))

# Everything in ticks: sigma in ticks/sqrt(s), kappa in 1/ticks.
print(quotes(mid=10_000.0, inventory=50, time_to_go=0.5, params=ASParams(gamma=0.001, sigma=2.0, kappa=0.5)))
```

## Repository guide

| Path | Contents |
|---|---|
| `src/microstructure/book.py` | Price-time priority order book |
| `src/microstructure/hawkes.py` | Multivariate exponential Hawkes: validated params, Ogata thinning, exact likelihood |
| `src/microstructure/avellaneda_stoikov.py` | Closed-form reservation price, spread and quotes, with a units contract |
| `src/microstructure/lobster.py` | LOBSTER message/orderbook parser |
| `tests/` | One test module per source module; `fixtures/` holds hand-made data files |
| `docs/MOTIVATION.md` | Why the project exists, why Hawkes, why the comparison is designed this way |
| `docs/ARCHITECTURE.md` | Layering rules, module contracts, event alphabet, design decisions |
| `docs/ROADMAP.md` | Phased plan with tests-first acceptance criteria |
| `legacy/cmu_ml2_meta_dqn/` | Earlier prototype of M5 (CMU ML2 Meta-DQN on SPY), superseded; its defects shaped this design |

## Data

No market data is committed: licensed or multi-gigabyte files live outside
the repository, and CI runs on hand-made fixtures in the same formats.

- **Nasdaq TotalView-ITCH 5.0.** Public full-day sample files from
  `emi.nasdaq.com/ITCH/`. This is the primary order-level source.
- **LOBSTER.** Message and orderbook CSVs. An account is required, and the
  parser is fixture-tested.
- **Databento MBP-10.** Top-10 price-level snapshots and trades, used as a
  cross-era check (see the roadmap for why it cannot support exact replay).

## License

MIT © 2026 Caleb Vinson

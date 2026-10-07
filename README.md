# Market Microstructure

[![ci](https://github.com/vincal848/MarketMicrostructure/actions/workflows/ci.yml/badge.svg)](https://github.com/vincal848/MarketMicrostructure/actions/workflows/ci.yml)

**A Hawkes-driven limit order book simulator for evaluating market makers,
calibrated to and validated against real Nasdaq order flow.**

Historical data cannot tell you how a market maker would have done. Its
orders would have changed the book, its queue position is unknown, and the
fills it would have received are exactly the ones followed by adverse price
moves. This project builds the standard alternative:

1. Reconstruct real order books message by message from Nasdaq
   TotalView-ITCH.
2. Classify the flow into six event types and fit a multivariate Hawkes
   process with multi-timescale kernels.
3. Simulate that flow forward, with market makers quoting into the same
   book.
4. Compare closed-form (Avellaneda-Stoikov) and learned (Double DQN) market
   makers on identical simulated flow. PnL is split into spread capture,
   adverse selection and inventory.

The full argument, with references, is in
**[docs/MOTIVATION.md](docs/MOTIVATION.md)**. The plan and its acceptance
criteria are in **[docs/ROADMAP.md](docs/ROADMAP.md)**, the real-data
results in **[docs/RESULTS.md](docs/RESULTS.md)**, and every change, tests
first, in **[CHANGELOG.md](CHANGELOG.md)**.

## Status

All seven roadmap phases are implemented, test-first. The suite has 226
tests at 97% coverage and passes `mypy --strict` and ruff. CI enforces the
layer architecture.

| Milestone | Acceptance | Result |
|---|---|---|
| M1 replay | Full ITCH day: zero unknown ids, zero price-time violations | **met**: 130,532 SPY executions audited, 0 violations |
| M2 estimation | MLE recovers simulated parameters within 4 SE; KS passes | **met** (CI), including six-type flow with tied events |
| M3 calibration | Hawkes beats Poisson on real flow; fit diagnostics reported | **met**: see [RESULTS.md](docs/RESULTS.md) |
| M3 validation | Simulated rates match theory; stylized facts compared with the real tape | see [RESULTS.md](docs/RESULTS.md) |
| M4 market makers | Calibrated baselines trade; PnL attributed, CIs over ≥30 seeds | see [RESULTS.md](docs/RESULTS.md) |
| M5 RL agent | Learned policy against the baselines on held-out seeds | see [RESULTS.md](docs/RESULTS.md) |

## Quick start

```bash
pip install -e ".[dev]"          # add ,rl for the PyTorch market maker
pytest -q                        # 226 tests
ruff check . && mypy             # lint, strict types
pre-commit install               # run both on every commit
```

The real-data pipeline. Nasdaq publishes free full-day ITCH sample files at
`emi.nasdaq.com/ITCH/`.

```bash
microstructure replay-itch    DAY.gz --symbol SPY --out runs/replay.json            # M1 audit
microstructure calibrate-itch DAY.gz --symbol SPY --out runs/cal.json \
    --marks-out runs/marks.npz                                                      # M3 fits, per 30 min
microstructure depth-itch     DAY.gz --symbol SPY --at 11:00 --out runs/depth.json  # opening book
microstructure stylized-itch  DAY.gz --symbol SPY --start 11:00 --end 11:30 --out runs/real.json
microstructure simulate   configs/spy_20190130_1100.toml --seeds 0:5 --out runs/sim.json
microstructure experiment configs/spy_20190130_1100.toml --out runs/                # M4 + M5
microstructure bench --out runs/bench.json
```

As a library:

```python
from microstructure.book import OrderBook, Side
from microstructure.hawkes import HawkesParams, simulate
from microstructure.hawkes_estimation import fit

book = OrderBook()
book.add_limit_order(1, Side.ASK, price=10_001, qty=100)
fills, leftover = book.market_order(Side.BID, 60)

truth = HawkesParams(mu=[0.3, 0.2], alpha=[[0.4, 0.1], [0.2, 0.3]], beta=[[1.0, 1.0], [1.0, 1.0]])
stream = simulate(truth, horizon=5_000.0, seed=0)
fitted = fit(stream, n_types=2, decay=1.0)
print(fitted.params.alpha, fitted.alpha_se, fitted.kkt_residual)
```

## Repository guide

| Path | Contents |
|---|---|
| `src/microstructure/` | The package: 21 modules in 8 layers ([ARCHITECTURE.md](docs/ARCHITECTURE.md)) |
| `tests/` | One test module per source module, plus property-based, architecture and real-data (`-m data`) tests |
| `configs/` | Experiment TOML files |
| `docs/MOTIVATION.md` | Why the project exists, why Hawkes, why the comparison is designed this way |
| `docs/ARCHITECTURE.md` | Layers, module contracts, event alphabet, design decisions |
| `docs/ROADMAP.md` | The phased plan with tests-first acceptance criteria |
| `docs/RESULTS.md` | Real-data acceptance numbers and the commands that produce them |
| `CHANGELOG.md` | Every phase: tests first, then implementation, then the defects found |
| `legacy/cmu_ml2_meta_dqn/` | The earlier CMU prototype whose defects shaped this design (superseded) |

## Data

No market data is committed: licensed or multi-gigabyte files live outside
the repository, and CI runs on hand-made fixtures in the same formats.

- **Nasdaq TotalView-ITCH 5.0.** Public full-day sample files. This is the
  primary, order-level source.
- **LOBSTER.** Message and orderbook CSVs. Fixture-tested, including
  windows that open with orders already resting.
- **Databento MBP-10.** Top-10 price levels plus trades, classified by
  snapshot diffs. It cannot be replayed at order level (see the roadmap).

## License

MIT © 2026 Caleb Vinson

# Results

Acceptance numbers measured on real data, with the command that produced
each one. Real data is not in the repository (see README > Data), so these
runs are local. Every number here is reproducible from the listed command
given the same input file.

## M1: exact order-level replay (Phase 1)

**Input.** Nasdaq TotalView-ITCH 5.0 sample day `01302019.NASDAQ_ITCH50.gz`
(4,764,426,091 bytes, MD5 `54b7afd0be7f925f7e9d67a23898827b`), symbol SPY.

**Command.**

```bash
microstructure replay-itch /path/to/01302019.NASDAQ_ITCH50.gz --symbol SPY \
    --out runs/m1_replay_spy_20190130.json
```

| Measure | Value |
|---|---|
| SPY order events replayed (full day, 04:00–20:00) | 3,065,003 |
| Visible executions audited (09:30–16:00) | 130,532 |
| Price-time priority violations | **0** |
| References to unknown order ids | **0** |
| Cancels/executions larger than the resting order | **0** |
| Displayed adds that would have crossed the book | **0** |
| Executions with an explicit price (crosses, not audited) | 4,637 |
| Hidden (non-displayed) executions | 9,041 |
| Book after the close | empty |
| Wall time for the whole file (all symbols decoded or skipped) | 230.5 s |

**Reading.** Every one of the 130,532 visible executions during continuous
trading hit the order the engine had first in line at the best price. So
the book reproduces Nasdaq's matching order exactly on a full day of a very
liquid symbol, and the depth it produces can be trusted as the base for
classification and calibration. The run took 230 s on one core. Most of
that is decompressing and skipping the other ~8,500 symbols' messages;
filtering one symbol costs one two-byte read per message.

**Acceptance test.** `MICROSTRUCTURE_ITCH=... pytest -m data` runs the same
replay and asserts every count above that should be zero is zero.

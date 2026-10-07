# CMU ML2 project: Meta-DQN market maker on SPY (SUPERSEDED)

**SUPERSEDED** by milestone M5 of this repository (an RL agent quoting
against simulated Hawkes flow, compared against the M4 baselines). Kept as a
record of the earlier prototype. It is not run in CI, and the notebooks do
not run as they are (see *Running*).

Course project for Machine Learning II at CMU, January–February 2026. A
Double DQN market maker quotes into a simulated fill model driven by real
order-book features. A second, "meta" variant adds an LSTM that watches the
recent market regime and retunes the reward's penalty weights while the
agent trades.

## Data

Databento `XNAS.ITCH` MBP-10 (top 10 levels) for **SPY**, 61 trading days
from 2025-11-11 to 2026-02-09. Each row was downsampled 1-in-100 and turned
into 23 features: mid price, microprice, spread, L1 and 5-level imbalance,
5-level VWAP and depth, book slope, 500-tick volatility, Hawkes intensity of
market events, add/cancel rates, order intensity and price change. The data
was split by day into train, validation and test sets of 350,235 / 161,351 /
158,605 rows. The raw data is licensed and not included.

## Method

| Notebook | Contents |
|---|---|
| `AMMCalc.ipynb` | `FeatureEngine`, `AMMEnvironment`, `DQNetwork`, `LSTMMetaLearner`, `MetaDQNAgent`, `AvellanedaStoikov` |
| `AMMData2.ipynb` | Per-day chunked loading, feature extraction and train/val/test split |
| `AMMFullTrain.ipynb` | Training, evaluation, summary table and plots (outputs kept) |

- **Environment.** Each episode is 1,000 steps, starting from $100k with a
  ±500 share inventory cap. There are 20 discrete actions, each a pair of
  quote offsets and a size (100/200/500), skewed by inventory. Fills are
  random, with a probability that rises with order-book imbalance and Hawkes
  intensity and falls exponentially with the quote's distance in ticks from
  the best price. Reward = change in wealth − λ0·|inventory| − λ1·inventory²
  − λ2·toxicity − fees.
- **DQN.** Double DQN with a 256-256-128 MLP over the 24-dimensional state
  (23 features plus normalised inventory), trained for 2,000 episodes with
  the penalty weights λ fixed.
- **Meta-DQN.** Same agent, but an LSTM maps 17 regime statistics from the
  last 100 states to λ every 50 steps. It is fitted to the λ settings that
  produced above-median PnL. Training was 500 warm-up episodes, then 1,500
  episodes with the LSTM active.
- **Baseline.** Avellaneda-Stoikov reservation price and optimal spread,
  with quotes kept at or behind the best bid and ask.

## Results

50 evaluation episodes per model and split (`AMMFullTrain.ipynb`, cell 9):

| Model | Split | PnL (mean ± std) | Sharpe | Trades | Final inv. |
|---|---|---|---|---|---|
| DQN | val | 981.49 ± 354.59 | 1.29 | 161.5 | −48.6 |
| DQN | test | 576.77 ± 134.72 | 1.38 | 161.2 | 34.5 |
| Meta-DQN | val | 1035.39 ± 339.27 | 1.35 | 161.1 | −1.0 |
| Meta-DQN | test | 591.23 ± 135.16 | 1.48 | 162.5 | 32.6 |
| Avellaneda-Stoikov | val | 29.73 ± 43.71 | 0.28 | 0.4 | −20.7 |
| Avellaneda-Stoikov | test | 0.00 ± 0.00 | 0.00 | 0.0 | 0.0 |

Plots and the full training and evaluation logs are in `results/`.

## Caveats

These were found when the project was archived, and they limit what the
table shows:

- **All 50 evaluation runs use the same window.** `evaluateEpisode` calls
  `env.reset()` with no arguments, so every run starts at index 0 and
  replays the first 1,000 steps of the split. The spread across runs comes
  only from the random fills, not from different market conditions.
- **The baseline is not a fair comparison.** `kappa=1.5` was given in
  dollars, which makes the optimal half-spread about $0.65, roughly 65 ticks
  behind the best price. Under the fill model's distance decay, that almost
  never fills, so Avellaneda-Stoikov barely trades. The DQN's lead over it
  says nothing about Avellaneda-Stoikov as a method.
- **PnL comes from the simulated fills, not from the real queue.** The fill
  probabilities are hand-set (`baseProb`, `tickDecay`, `aggressiveBonus`
  and so on), so the absolute PnL depends on those choices.
- **The meta variant's lead is small.** Meta-DQN beats DQN by about 15–55
  in mean PnL, against standard deviations of 135–355, under the
  single-window setup described above.

M4 and M5 of this repository address these directly. Those milestones use
generative Hawkes flow instead of a hand-set fill model, compare against a
calibrated baseline under identical flow, and split PnL into spread capture
and adverse selection.

## Running

The notebooks load each other with `%run C:/Users/caleb/ML2Project/...` and
read raw data from `D:\MLHFTData\DecompressedData`. Both paths are from the
original machine. To rerun, point the `%run` lines at this folder and
`RAW_DIR` at a local copy of the Databento files.

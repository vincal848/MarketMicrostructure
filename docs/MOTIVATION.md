# Motivation

This document explains why the project exists, which problem it attacks, why
the chosen modelling approach is the right one for that problem, and what the
earlier attempt (`legacy/cmu_ml2_meta_dqn/`) taught about how not to do it.

## 1. The problem: you cannot backtest a market maker on historical data

A market maker continuously posts a bid and an ask. It earns the spread when
both sides fill against uninformed flow, and it loses when it is filled by
traders who know where the price is going next, or when it is left holding
inventory as the price moves. Those two losses are the classical costs of
liquidity provision:

- **Adverse selection.** Glosten & Milgrom (1985) and Kyle (1985) show that
  quotes must be wide enough to cover losses to informed traders. A fill is
  not random: it is more likely precisely when the price is about to move
  through the quote.
- **Inventory risk.** Ho & Stoll (1981) and Avellaneda & Stoikov (2008) model
  a dealer who must skew quotes to shed inventory, trading spread income for
  lower variance.

Evaluating a market-making strategy means answering a *counterfactual*
question: what would the book, the fills, and the subsequent prices have been
if this agent had been quoting? Historical data cannot answer it, for three
reasons:

1. **Queue position.** Whether a passive order fills depends on how many
   shares were ahead of it at the same price. Replaying history does not tell
   you where your order would have sat, and naive fill rules ("filled if the
   price traded through") systematically overstate fill rates.
2. **Market impact and reaction.** The agent's own orders change the book.
   Other participants react: cancellations follow aggressive trades,
   replenishment follows depletion. A replay holds the rest of the market fixed
   and so cannot express this.
3. **Adverse selection is endogenous.** The fills a market maker receives
   are correlated with the flow that moves the price. A fill model that draws
   fills independently of future price moves (as the legacy project did)
   removes the dominant cost of market making and produces flattering PnL.

The industry answer is a **simulator that is generative but anchored to
data**: it can be run forward under an agent's influence, yet its order flow
is calibrated to, and validated against, real message data.

## 2. Why order flow should be a multivariate Hawkes process

There are roughly three generations of limit order book simulators:

| Generation | Order flow | What it gets wrong |
|---|---|---|
| Historical replay | Exactly what happened | No counterfactuals: the agent cannot change anything |
| Zero-intelligence / Poisson (Smith, Farmer et al. 2003; Cont, Stoikov & Talreja 2010) | Independent Poisson arrivals per event type | No clustering: bursts of market orders and waves of cancellations, which drive adverse selection, do not occur |
| Self-exciting (Large 2007; Bacry, Mastromatteo & Muzy 2015) | Hawkes processes, where each event raises the near-term rate of others | Calibration and validation are harder, which is why this project spends three of its phases on them |

Empirically, order flow is strongly self- and cross-exciting. Market orders
arrive in bursts. A market sell raises the probability of further market
sells, of bid-side cancellations (market makers pulling quotes after toxic
flow), and of ask-side limit adds (replenishment). Hardiman, Bercot &
Bouchaud (2013) and Filimonov & Sornette (2012) estimate that a large share
of all events is endogenously triggered by earlier events rather than by
exogenous news. Their measure is the Hawkes branching ratio: the spectral
radius of the branching matrix, close to 1 in modern markets.

A multivariate Hawkes process with exponential kernels captures this with a
small, interpretable parameter set:

```
lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} alpha_ij * exp(-beta_ij (t - t_k^j))
```

- `mu_i` is the exogenous rate of event type `i`.
- `alpha_ij / beta_ij` is the expected number of type-`i` events directly
  triggered by one type-`j` event.
- The off-diagonal structure of `alpha` is exactly the "market sell triggers
  bid cancels" mechanism that makes fills adversely selected.

It also has properties that matter in practice:

- **Exact likelihood.** Maximum likelihood is principled, and for a fixed
  decay the likelihood is concave in `(mu, alpha)`, so estimation is reliable.
- **Testable fit.** The time-rescaling theorem (Brown et al. 2002) turns a
  fitted model into a goodness-of-fit test: the compensator increments
  between events must be i.i.d. Exp(1).
- **Exact simulation.** Ogata's (1981) thinning simulates the process
  exactly, and it can be run *online*, so an agent's own market orders excite
  the background flow. That is market impact arising from the model rather
  than being bolted on.

Queue-reactive models (Huang, Lehalle & Rosenbaum 2015) are the main
alternative. They condition intensities on the current queue sizes instead
of on event history. They are a natural extension once the Hawkes baseline is
validated (see [ROADMAP.md](ROADMAP.md), "Beyond the plan").

## 3. Why compare a learned agent against closed-form market makers

Avellaneda-Stoikov gives an optimal policy, but only under assumptions the
simulator deliberately violates:

- Brownian mid-price.
- Poisson fills whose intensity depends only on distance from mid.
- No adverse selection.
- No queue.

That gap is the research question. **When order flow clusters and fills are
adversely selected, how much does a policy that observes the order book
(imbalance, recent flow, its own queue position) gain over the closed-form
quotes, and where does that gain come from?** Reinforcement learning is the
standard tool for policies without a closed form (Spooner et al. 2018;
Cartea, Jaimungal & Penalva 2015 for the stochastic-control side).

A positive answer is only meaningful if it survives three controls:

1. **A correctly calibrated baseline.** The Avellaneda-Stoikov parameters
   `sigma` and `kappa` must be estimated from the same simulator, in the same
   price units, as the agent sees.
2. **Identical flow.** Every policy is evaluated on the same random seeds
   (common random numbers), so differences are paired, not confounded by luck.
3. **PnL attribution.** PnL is split into spread capture, adverse selection
   and inventory PnL. "Earned more" then becomes "earned more spread", "was
   picked off less", or "got lucky on inventory".

## 4. What the legacy project taught

`legacy/cmu_ml2_meta_dqn/` is a Meta-DQN market maker trained on 61 days of
SPY data (CMU Machine Learning II, early 2026). It reported a Sharpe ratio of
about 1.4 against an Avellaneda-Stoikov baseline that made almost no trades.
Archiving it surfaced the failure modes this repository is structured to
prevent:

| Legacy defect | Consequence | Where this repository prevents it |
|---|---|---|
| Fills drawn from a hand-set probability, independent of future prices | No adverse selection, so PnL is flattering | Fills come from matching against simulated order flow (Phases 4–5) |
| `kappa` given in dollars while quotes were clipped to the touch | The baseline sat about 65 ticks away and never filled | Units contract and rescaling test in `avellaneda_stoikov.py`; `kappa` estimated from the simulator (Phase 5) |
| Every evaluation episode replayed the same window | 50 "runs" were 1 window × 50 fill-RNG draws | Seeded, paired, held-out evaluation with confidence intervals (Phases 5–6) |
| Features taken from the historical book while the agent's orders were invisible to it | The agent could not affect the market | The agent's orders live in the same book as the simulated flow (Phase 4) |
| Notebook pipeline linked by `%run` against absolute paths | Not reproducible, not testable | Typed package, CI, configs and run manifests (Phase 7) |

The legacy features are kept as a reference set for the RL observation
space: microprice, imbalance, depth slope and Hawkes intensity.

## 5. Scope and non-goals

In scope:

- One instrument at a time.
- Displayed limit order book with price-time priority.
- Nasdaq TotalView-ITCH and LOBSTER as order-level data sources.
- Databento MBP-10 as a level-aggregated cross-check.
- Market makers that quote at most one order per side.

Not in scope:

- Hidden and iceberg liquidity, beyond counting hidden executions as trades.
- Auctions (open/close crosses are excluded from calibration windows).
- Multi-venue routing.
- Exchange fee schedules beyond a flat maker/taker fee.
- Live trading.
- Microsecond-accurate latency modelling. Latency is a configurable delay,
  not a network model.

## References

- Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance* 8(3).
- Bacry, E., Mastromatteo, I. & Muzy, J.-F. (2015). Hawkes processes in finance. *Market Microstructure and Liquidity* 1(1).
- Bouchaud, J.-P., Bonart, J., Donier, J. & Gould, M. (2018). *Trades, Quotes and Prices*. Cambridge University Press.
- Brown, E. N., Barbieri, R., Ventura, V., Kass, R. E. & Frank, L. M. (2002). The time-rescaling theorem and its application to neural spike train data analysis. *Neural Computation* 14(2).
- Cartea, Á., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
- Cont, R., Stoikov, S. & Talreja, R. (2010). A stochastic model for order book dynamics. *Operations Research* 58(3).
- Filimonov, V. & Sornette, D. (2012). Quantifying reflexivity in financial markets. *Physical Review E* 85.
- Glosten, L. & Milgrom, P. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics* 14(1).
- Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2013). Dealing with the inventory risk. *Mathematics and Financial Economics* 7(4).
- Hardiman, S., Bercot, N. & Bouchaud, J.-P. (2013). Critical reflexivity in financial markets: a Hawkes process analysis. *European Physical Journal B* 86.
- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika* 58(1).
- Hawkes, A. G. & Oakes, D. (1974). A cluster process representation of a self-exciting process. *Journal of Applied Probability* 11(3).
- Ho, T. & Stoll, H. (1981). Optimal dealer pricing under transactions and return uncertainty. *Journal of Financial Economics* 9(1).
- Huang, W., Lehalle, C.-A. & Rosenbaum, M. (2015). Simulating and analyzing order book data: the queue-reactive model. *Journal of the American Statistical Association* 110.
- Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica* 53(6).
- Large, J. (2007). Measuring the resiliency of an electronic limit order book. *Journal of Financial Markets* 10(1).
- Ogata, Y. (1981). On Lewis' simulation method for point processes. *IEEE Transactions on Information Theory* 27(1).
- Ozaki, T. (1979). Maximum likelihood estimation of Hawkes' self-exciting point processes. *Annals of the Institute of Statistical Mathematics* 31.
- Smith, E., Farmer, J. D., Gillemot, L. & Krishnamurthy, S. (2003). Statistical theory of the continuous double auction. *Quantitative Finance* 3(6).
- Spooner, T., Fearnley, J., Savani, R. & Koukorinis, A. (2018). Market making via reinforcement learning. *Proceedings of AAMAS 2018*.

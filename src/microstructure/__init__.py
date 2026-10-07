"""Hawkes-driven limit order book simulation for evaluating market makers.

Layers, bottom to top (docs/ARCHITECTURE.md; enforced by tests/test_architecture.py):

    0  events, book, hawkes, hawkes_estimation, avellaneda_stoikov   pure core
    1  itch, lobster                                                 order-level sources
    2  replay                                                        book replay and audit
    3  flow, calibration, databento, stylized                        flow, fits, tape statistics
    4  simulator                                                     discrete-event market
    5  agents, accounting                                            market makers, PnL
    6  evaluation, env, rl                                           experiments
    7  experiment, bench, cli                                        entry points
"""

__version__ = "1.0.0"

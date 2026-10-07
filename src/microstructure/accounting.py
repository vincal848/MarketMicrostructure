"""PnL ledger and the industry-standard attribution of market-making PnL.

For a fill of `q` shares at price `p` in direction `d` (+1 bought, -1 sold),
with mid `m0` at the fill, `m_h` a horizon `h` later and `m_T` at the end:

    spread capture      d (m0  - p)   q    earned against the mid at the fill
    adverse selection   d (m_h - m0)  q    how the mid moved right after; negative
                                           when informed flow picked the quote off
    inventory           d (m_T - m_h) q    what holding the position did after that

Summed over fills, these give exactly the mark-to-market PnL before fees,
since d (m_T - p) q splits into the three terms above. The split
says *why* a strategy made or lost money: wide quotes earn spread capture,
toxic fills show up as adverse selection, and luck on the position shows
up as inventory PnL. This is the realized-spread decomposition used in
execution-quality reporting (e.g. SEC Rule 605) applied fill by fill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from microstructure.book import Side
from microstructure.hawkes import FloatArray
from microstructure.simulator import AgentFill
from microstructure.stylized import MarketTape


@dataclass(frozen=True)
class MidPath:
    """Mid prices observed at increasing `times`; the mid at time t is the
    last observation at or before t (the first one before any)."""

    times: FloatArray
    mids: FloatArray

    @classmethod
    def from_tape(cls, tape: MarketTape) -> MidPath:
        return cls(tape.times, tape.mid)

    def at(self, times: FloatArray) -> FloatArray:
        index = np.clip(np.searchsorted(self.times, times, side="right") - 1, 0, len(self.mids) - 1)
        mids: FloatArray = self.mids[index]
        return mids


@dataclass(frozen=True)
class PnLAttribution:
    spread_capture: float
    adverse_selection: float
    inventory: float
    fees: float
    total: float  # = spread_capture + adverse_selection + inventory - fees


class Ledger:
    """Cash, inventory and fees from a sequence of fills.

    Fees are per share in price units: positive is a charge, negative a
    rebate. Passive fills pay `maker_fee`, aggressive ones `taker_fee`. Cash
    excludes fees, which are tracked separately.
    """

    def __init__(self, maker_fee: float = 0.0, taker_fee: float = 0.0) -> None:
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.fills: list[AgentFill] = []
        self.cash = 0.0
        self.inventory = 0
        self.fees = 0.0
        self.volume = 0
        self.max_abs_inventory = 0

    def record(self, fill: AgentFill) -> None:
        direction = 1 if fill.side is Side.BID else -1
        self.fills.append(fill)
        self.cash -= direction * fill.price * fill.qty
        self.inventory += direction * fill.qty
        self.fees += (self.taker_fee if fill.aggressive else self.maker_fee) * fill.qty
        self.volume += fill.qty
        self.max_abs_inventory = max(self.max_abs_inventory, abs(self.inventory))

    def mark_to_market(self, mid: float) -> float:
        return self.cash + self.inventory * mid - self.fees

    def attribution(self, path: MidPath, horizon: float, final_mid: float) -> PnLAttribution:
        """Split PnL into spread capture, adverse selection at `horizon`
        seconds, and inventory, valuing the final position at `final_mid`."""
        if not self.fills:
            return PnLAttribution(0.0, 0.0, 0.0, self.fees, -self.fees)
        times = np.array([f.time for f in self.fills])
        direction = np.array([1.0 if f.side is Side.BID else -1.0 for f in self.fills])
        price = np.array([f.price for f in self.fills], dtype=np.float64)
        qty = np.array([f.qty for f in self.fills], dtype=np.float64)
        observed = path.at(times)
        mid_at_fill = np.array([observed[k] if f.mid is None else f.mid for k, f in enumerate(self.fills)])
        mid_after = path.at(times + horizon)
        signed = direction * qty
        spread = float(np.sum(signed * (mid_at_fill - price)))
        adverse = float(np.sum(signed * (mid_after - mid_at_fill)))
        inventory = float(np.sum(signed * (final_mid - mid_after)))
        return PnLAttribution(spread, adverse, inventory, self.fees, self.mark_to_market(final_mid))

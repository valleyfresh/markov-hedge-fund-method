from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from markov.engine import Regime


@dataclass
class SignalEvent:
    ticker: str
    action: Literal["LONG_ENTRY", "SHORT_ENTRY", "LONG_EXIT", "SHORT_EXIT"]
    conviction: float
    conviction_tier: Literal["LOW", "MEDIUM", "HIGH"]
    risk_pct: float
    spy_regime: Regime
    ticker_regime: Regime


_TIERS = [
    (0.00, 0.40, "LOW",    0.01),
    (0.40, 0.70, "MEDIUM", 0.02),
    (0.70, math.inf, "HIGH", 0.03),
]


def _tier(conviction: float) -> tuple[str, float]:
    if not 0.0 <= conviction <= 1.0:
        raise ValueError(f"conviction must be in [0, 1], got {conviction}")
    for low, high, label, pct in _TIERS:
        if low <= conviction < high:
            return label, pct
    raise AssertionError(f"_tier fallthrough for conviction={conviction}")


class SignalRouter:
    """Stateless per-bar router. Caller manages position state."""

    def evaluate(
        self,
        ticker: str,
        ticker_regime: Regime,
        spy_regime: Regime,
        current_side: Literal["flat", "long", "short"],
        conviction: float,
    ) -> SignalEvent | None:
        if current_side not in {"flat", "long", "short"}:
            raise ValueError(f"Unknown current_side: {current_side!r}")
        tier_label, risk_pct = _tier(conviction)

        def _event(action: str) -> SignalEvent:
            return SignalEvent(
                ticker=ticker,
                action=action,
                conviction=conviction,
                conviction_tier=tier_label,
                risk_pct=risk_pct,
                spy_regime=spy_regime,
                ticker_regime=ticker_regime,
            )

        if current_side == "long":
            if ticker_regime == Regime.Bear or spy_regime == Regime.Bear:
                return _event("LONG_EXIT")
            return None

        if current_side == "short":
            if ticker_regime == Regime.Bull or spy_regime == Regime.Bull:
                return _event("SHORT_EXIT")
            return None

        if ticker_regime == Regime.Bull and spy_regime != Regime.Bear:
            return _event("LONG_ENTRY")
        if ticker_regime == Regime.Bear and spy_regime != Regime.Bull:
            return _event("SHORT_ENTRY")
        return None

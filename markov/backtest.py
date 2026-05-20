from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Trade:
    ticker: str
    side: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_price: float
    conviction: float
    risk_pct: float
    shares: float
    gross_return: float
    net_return: float


@dataclass
class BacktestResult:
    ticker: str
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0


class WalkForwardBacktester:
    def __init__(self, config: dict) -> None:
        self.lookback_days    = config["lookback_days"]
        self.threshold_pct    = config["threshold_pct"]
        self.train_days       = config["backtest"]["train_days"]
        self.test_days        = config["backtest"]["test_days"]
        self.tx_cost          = config["backtest"]["tx_cost_pct"]
        self.slippage         = config["backtest"]["slippage_pct"]
        self.short_borrow     = config["backtest"]["short_borrow_daily_pct"]
        self.portfolio_value  = config["risk"]["portfolio_value"]
        self.atr_period       = config["risk"]["atr_period"]
        self.atr_multiplier   = config["risk"]["atr_multiplier"]
        self.conviction_tiers = config["risk"]["conviction_tiers"]

    def _risk_pct(self, conviction: float) -> float:
        for tier in self.conviction_tiers.values():
            low, high, pct = tier
            if low <= conviction < high:
                return float(pct)
        return float(self.conviction_tiers["high"][2])

    def _position_size(
        self,
        portfolio_value: float,
        risk_pct: float,
        entry_price: float,
        stop_price: float,
    ) -> float:
        stop_distance = abs(entry_price - stop_price)
        if stop_distance == 0:
            return 0.0
        return (portfolio_value * risk_pct) / stop_distance

    def _atr(self, ohlcv: pd.DataFrame, period: int) -> pd.Series:
        high  = ohlcv["High"]
        low   = ohlcv["Low"]
        close = ohlcv["Close"]
        prev  = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
        ).max(axis=1)
        return tr.rolling(period).mean()

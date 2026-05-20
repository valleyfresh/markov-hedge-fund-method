import pandas as pd
import pytest
from markov.backtest import BacktestResult, Trade


def _make_trade(side: str = "long", net_return: float = 0.05, risk_pct: float = 0.01) -> Trade:
    return Trade(
        ticker="SPY",
        side=side,
        entry_date=pd.Timestamp("2024-01-02"),
        exit_date=pd.Timestamp("2024-01-15"),
        entry_price=470.0,
        exit_price=494.0 if side == "long" else 447.0,
        stop_price=455.0 if side == "long" else 485.0,
        conviction=0.65,
        risk_pct=risk_pct,
        shares=12.5,
        gross_return=0.051,
        net_return=net_return,
    )


def test_trade_fields_accessible():
    t = _make_trade()
    assert t.ticker == "SPY"
    assert t.side == "long"
    assert t.net_return == 0.05


def test_backtest_result_defaults_to_empty():
    r = BacktestResult(ticker="SPY")
    assert r.trades == []
    assert r.sharpe == 0.0
    assert r.max_drawdown == 0.0
    assert r.win_rate == 0.0

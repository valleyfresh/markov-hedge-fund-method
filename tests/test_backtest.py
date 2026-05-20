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


import yaml
from markov.backtest import WalkForwardBacktester


def _cfg() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def test_conviction_tier_low():
    bt = WalkForwardBacktester(_cfg())
    assert bt._risk_pct(0.20) == pytest.approx(0.01)


def test_conviction_tier_medium():
    bt = WalkForwardBacktester(_cfg())
    assert bt._risk_pct(0.55) == pytest.approx(0.02)


def test_conviction_tier_high():
    bt = WalkForwardBacktester(_cfg())
    assert bt._risk_pct(0.75) == pytest.approx(0.03)


def test_position_size_formula():
    bt = WalkForwardBacktester(_cfg())
    shares = bt._position_size(
        portfolio_value=10000,
        risk_pct=0.01,
        entry_price=470.0,
        stop_price=455.0,
    )
    assert shares == pytest.approx(100.0 / 15.0)


def test_position_size_zero_when_stop_at_entry():
    bt = WalkForwardBacktester(_cfg())
    shares = bt._position_size(10000, 0.01, 470.0, 470.0)
    assert shares == 0.0


def test_atr_computed_correctly():
    bt = WalkForwardBacktester(_cfg())
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    ohlcv = pd.DataFrame({
        "Open":  [100.0] * n,
        "High":  [101.0] * n,
        "Low":   [99.0]  * n,
        "Close": [100.0] * n,
        "Volume":[1_000_000] * n,
    }, index=idx)
    atr = bt._atr(ohlcv, period=14)
    assert atr.iloc[-1] == pytest.approx(2.0, rel=1e-3)


from markov.engine import Regime


def test_spy_bull_suppresses_short():
    bt = WalkForwardBacktester(_cfg())
    assert bt._apply_gate(ticker_regime=Regime.Bear, spy_regime=Regime.Bull, side="short") is False


def test_spy_bear_suppresses_long():
    bt = WalkForwardBacktester(_cfg())
    assert bt._apply_gate(ticker_regime=Regime.Bull, spy_regime=Regime.Bear, side="long") is False


def test_spy_sideways_allows_long():
    bt = WalkForwardBacktester(_cfg())
    assert bt._apply_gate(ticker_regime=Regime.Bull, spy_regime=Regime.Sideways, side="long") is True


def test_spy_sideways_allows_short():
    bt = WalkForwardBacktester(_cfg())
    assert bt._apply_gate(ticker_regime=Regime.Bear, spy_regime=Regime.Sideways, side="short") is True


def test_spy_bull_allows_long():
    bt = WalkForwardBacktester(_cfg())
    assert bt._apply_gate(ticker_regime=Regime.Bull, spy_regime=Regime.Bull, side="long") is True


def test_spy_bear_allows_short():
    bt = WalkForwardBacktester(_cfg())
    assert bt._apply_gate(ticker_regime=Regime.Bear, spy_regime=Regime.Bear, side="short") is True

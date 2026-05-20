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


def test_long_net_return_deducts_costs():
    bt = WalkForwardBacktester(_cfg())
    # Long: buy at 100, sell at 110 → gross = 10/100 = 0.10
    # round-trip cost = 2 * (tx_cost + slippage) = 2 * (0.001 + 0.0005) = 0.003
    # net = 0.10 - 0.003 = 0.097
    net = bt._net_return(side="long", entry=100.0, exit_=110.0, days_held=5)
    assert net == pytest.approx(0.10 - 2 * (0.001 + 0.0005), rel=1e-6)


def test_short_net_return_deducts_costs_and_borrow():
    bt = WalkForwardBacktester(_cfg())
    # Short: sell at 100, buy back at 90 → gross = (100-90)/100 = 0.10
    # borrow cost = 0.0001 * 5 days = 0.0005
    # round-trip cost = 0.003
    # net = 0.10 - 0.003 - 0.0005 = 0.0965
    net = bt._net_return(side="short", entry=100.0, exit_=90.0, days_held=5)
    assert net == pytest.approx(0.10 - 2 * (0.001 + 0.0005) - 0.0001 * 5, rel=1e-6)


def test_long_losing_trade_net_return_is_negative():
    bt = WalkForwardBacktester(_cfg())
    net = bt._net_return(side="long", entry=100.0, exit_=95.0, days_held=3)
    assert net < 0


def test_short_losing_trade_net_return_is_negative():
    bt = WalkForwardBacktester(_cfg())
    net = bt._net_return(side="short", entry=100.0, exit_=105.0, days_held=3)
    assert net < 0


def test_metrics_win_rate():
    bt = WalkForwardBacktester(_cfg())
    trades = [
        _make_trade(net_return=0.05),
        _make_trade(net_return=-0.02),
        _make_trade(net_return=0.03),
    ]
    result = bt._compute_metrics("SPY", trades, 10150.0)
    assert result.win_rate == pytest.approx(2 / 3)


def test_metrics_avg_win_avg_loss():
    bt = WalkForwardBacktester(_cfg())
    trades = [
        _make_trade(net_return=0.06),
        _make_trade(net_return=0.04),
        _make_trade(net_return=-0.02),
    ]
    result = bt._compute_metrics("SPY", trades, 10200.0)
    assert result.avg_win  == pytest.approx(0.05)
    assert result.avg_loss == pytest.approx(-0.02)


def test_metrics_max_drawdown_negative():
    bt = WalkForwardBacktester(_cfg())
    trades = [
        _make_trade(net_return=0.10),
        _make_trade(net_return=-0.20),
        _make_trade(net_return=0.05),
    ]
    result = bt._compute_metrics("SPY", trades, 9000.0)
    assert result.max_drawdown < 0


def test_metrics_empty_trades_returns_zeros():
    bt = WalkForwardBacktester(_cfg())
    result = bt._compute_metrics("SPY", [], 10000.0)
    assert result.sharpe == 0.0
    assert result.win_rate == 0.0
    assert result.trades == []

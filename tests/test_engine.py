import pandas as pd
import pytest
from markov.engine import Regime, RegimeEngine


def _prices(daily_pct: float, n: int = 60, start: float = 100.0) -> pd.Series:
    """Build a price series with a fixed daily return for n bars."""
    prices = [start * (1 + daily_pct) ** i for i in range(n)]
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx)


def _mixed_prices(n: int = 120) -> pd.Series:
    """Alternating up/down blocks to generate all three regimes."""
    block = n // 3
    up   = [100.0 * (1.01) ** i for i in range(block)]
    flat = [up[-1] * (1.001) ** i for i in range(block)]
    down = [flat[-1] * (0.99) ** i for i in range(block)]
    prices = up + flat + down
    idx = pd.date_range("2023-01-01", periods=len(prices), freq="B")
    return pd.Series(prices, index=idx)


def test_regime_enum_values():
    assert Regime.Bull == 0
    assert Regime.Bear == 1
    assert Regime.Sideways == 2


def test_current_regime_bull():
    # 1%/day for 60 bars → 20-day return ≈ 22% → above 5% threshold
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_prices(0.01))
    assert engine.current_regime() == Regime.Bull


def test_current_regime_bear():
    # -1%/day → 20-day return ≈ -18% → below -5% threshold
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_prices(-0.01))
    assert engine.current_regime() == Regime.Bear


def test_current_regime_sideways():
    # 0.1%/day → 20-day return ≈ 2% → within ±5% band
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_prices(0.001))
    assert engine.current_regime() == Regime.Sideways


def test_raises_before_fit():
    engine = RegimeEngine()
    with pytest.raises(RuntimeError, match="fit"):
        engine.current_regime()


def test_transition_matrix_rows_sum_to_one():
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_mixed_prices())
    for row in engine._matrix:
        assert abs(row.sum() - 1.0) < 1e-9


def test_signal_returns_regime_and_conviction_in_range():
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_prices(0.01))
    regime, conviction = engine.signal()
    assert isinstance(regime, Regime)
    assert 0.0 <= conviction <= 1.0


def test_signal_bull_when_bull_prob_exceeds_bear():
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_prices(0.01))
    regime, _ = engine.signal()
    assert regime == Regime.Bull


def test_signal_bear_when_bear_prob_exceeds_bull():
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_prices(-0.01))
    regime, _ = engine.signal()
    assert regime == Regime.Bear


def test_stationary_dist_sums_to_one():
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_mixed_prices())
    dist = engine.stationary_dist()
    assert set(dist.keys()) == {"Bull", "Bear", "Sideways"}
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    for v in dist.values():
        assert v >= 0.0


def test_n_step_forecast_sums_to_one():
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_mixed_prices())
    forecast = engine.n_step_forecast(5)
    assert set(forecast.keys()) == {"Bull", "Bear", "Sideways"}
    assert abs(sum(forecast.values()) - 1.0) < 1e-6


def test_n_step_forecast_n1_matches_matrix_row():
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_mixed_prices())
    forecast = engine.n_step_forecast(1)
    current = int(engine.current_regime())
    assert abs(forecast["Bull"]     - engine._matrix[current, Regime.Bull])     < 1e-9
    assert abs(forecast["Bear"]     - engine._matrix[current, Regime.Bear])     < 1e-9
    assert abs(forecast["Sideways"] - engine._matrix[current, Regime.Sideways]) < 1e-9

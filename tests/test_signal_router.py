import pytest
from markov.engine import Regime
from markov.signal_router import SignalEvent, SignalRouter


def _router() -> SignalRouter:
    return SignalRouter()


def test_long_entry_when_spy_bull_and_ticker_bull():
    router = _router()
    event = router.evaluate(
        ticker="AAPL",
        ticker_regime=Regime.Bull,
        spy_regime=Regime.Bull,
        current_side="flat",
        conviction=0.74,
    )
    assert event is not None
    assert event.action == "LONG_ENTRY"
    assert event.ticker == "AAPL"


def test_long_entry_when_spy_sideways_and_ticker_bull():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bull, Regime.Sideways, "flat", 0.55)
    assert event is not None
    assert event.action == "LONG_ENTRY"


def test_no_long_entry_when_spy_bear():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bull, Regime.Bear, "flat", 0.74)
    assert event is None


def test_short_entry_when_spy_bear_and_ticker_bear():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bear, Regime.Bear, "flat", 0.65)
    assert event is not None
    assert event.action == "SHORT_ENTRY"


def test_short_entry_when_spy_sideways_and_ticker_bear():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bear, Regime.Sideways, "flat", 0.50)
    assert event is not None
    assert event.action == "SHORT_ENTRY"


def test_no_short_entry_when_spy_bull():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bear, Regime.Bull, "flat", 0.65)
    assert event is None


def test_hold_when_long_in_bull_regime():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bull, Regime.Bull, "long", 0.80)
    assert event is None


def test_hold_when_short_in_bear_regime():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bear, Regime.Bear, "short", 0.80)
    assert event is None


def test_long_exit_when_ticker_turns_bear():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bear, Regime.Sideways, "long", 0.60)
    assert event is not None
    assert event.action == "LONG_EXIT"


def test_long_exit_when_spy_turns_bear():
    router = _router()
    event = router.evaluate("AAPL", Regime.Sideways, Regime.Bear, "long", 0.30)
    assert event is not None
    assert event.action == "LONG_EXIT"


def test_short_exit_when_ticker_turns_bull():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bull, Regime.Sideways, "short", 0.70)
    assert event is not None
    assert event.action == "SHORT_EXIT"


def test_short_exit_when_spy_turns_bull():
    router = _router()
    event = router.evaluate("AAPL", Regime.Sideways, Regime.Bull, "short", 0.40)
    assert event is not None
    assert event.action == "SHORT_EXIT"


def test_no_event_when_long_and_sideways():
    router = _router()
    event = router.evaluate("AAPL", Regime.Sideways, Regime.Sideways, "long", 0.20)
    assert event is None


def test_no_event_when_flat_and_sideways():
    router = _router()
    event = router.evaluate("AAPL", Regime.Sideways, Regime.Sideways, "flat", 0.20)
    assert event is None


def test_signal_event_conviction_tier_high():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bull, Regime.Bull, "flat", 0.75)
    assert event.conviction_tier == "HIGH"
    assert event.risk_pct == pytest.approx(0.03)


def test_signal_event_conviction_tier_medium():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bull, Regime.Bull, "flat", 0.50)
    assert event.conviction_tier == "MEDIUM"
    assert event.risk_pct == pytest.approx(0.02)


def test_signal_event_conviction_tier_low():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bull, Regime.Bull, "flat", 0.20)
    assert event.conviction_tier == "LOW"
    assert event.risk_pct == pytest.approx(0.01)


def test_signal_event_has_spy_and_ticker_regime():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bull, Regime.Sideways, "flat", 0.55)
    assert event.spy_regime == Regime.Sideways
    assert event.ticker_regime == Regime.Bull


def test_signal_event_conviction_tier_at_boundary_040():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bull, Regime.Bull, "flat", 0.40)
    assert event.conviction_tier == "MEDIUM"
    assert event.risk_pct == pytest.approx(0.02)


def test_signal_event_conviction_tier_at_boundary_070():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bull, Regime.Bull, "flat", 0.70)
    assert event.conviction_tier == "HIGH"
    assert event.risk_pct == pytest.approx(0.03)


def test_signal_event_conviction_tier_at_max_100():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bull, Regime.Bull, "flat", 1.0)
    assert event.conviction_tier == "HIGH"
    assert event.risk_pct == pytest.approx(0.03)


def test_invalid_conviction_raises():
    router = _router()
    with pytest.raises(ValueError, match="conviction"):
        router.evaluate("AAPL", Regime.Bull, Regime.Bull, "flat", 1.5)


def test_invalid_current_side_raises():
    router = _router()
    with pytest.raises(ValueError, match="current_side"):
        router.evaluate("AAPL", Regime.Bull, Regime.Bull, "FLAT", 0.5)

from unittest.mock import MagicMock, patch
import pytest
from markov.engine import Regime
from markov.notifier import TelegramNotifier
from markov.signal_router import SignalEvent


def _notifier() -> TelegramNotifier:
    return TelegramNotifier(token="test-token", chat_id="12345")


def _entry_event(action: str = "LONG_ENTRY") -> SignalEvent:
    return SignalEvent(
        ticker="AAPL",
        action=action,
        conviction=0.74,
        conviction_tier="HIGH",
        risk_pct=0.03,
        spy_regime=Regime.Sideways,
        ticker_regime=Regime.Bull,
    )


def test_send_calls_telegram_api():
    notifier = _notifier()
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(ok=True)
        notifier.send(_entry_event(), price=212.40, stop=205.10, shares=47)
    mock_post.assert_called_once()
    url = mock_post.call_args[0][0]
    assert "test-token" in url
    assert "sendMessage" in url


def test_send_payload_contains_ticker():
    notifier = _notifier()
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(ok=True)
        notifier.send(_entry_event(), price=212.40, stop=205.10, shares=47)
    payload = mock_post.call_args[1]["json"]
    assert "AAPL" in payload["text"]
    assert payload["chat_id"] == "12345"


def test_long_entry_message_contains_required_fields():
    notifier = _notifier()
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(ok=True)
        notifier.send(_entry_event("LONG_ENTRY"), price=212.40, stop=205.10, shares=47)
    text = mock_post.call_args[1]["json"]["text"]
    assert "LONG" in text
    assert "212" in text    # price
    assert "205" in text    # stop
    assert "47" in text     # shares
    assert "HIGH" in text   # conviction tier
    assert "3%" in text     # risk pct


def test_short_entry_message_contains_short_label():
    notifier = _notifier()
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(ok=True)
        notifier.send(_entry_event("SHORT_ENTRY"), price=212.40, stop=220.0, shares=30)
    text = mock_post.call_args[1]["json"]["text"]
    assert "SHORT" in text


def test_exit_message_contains_exit_label():
    notifier = _notifier()
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(ok=True)
        notifier.send(_entry_event("LONG_EXIT"), price=225.0, stop=0.0, shares=47)
    text = mock_post.call_args[1]["json"]["text"]
    assert "EXIT" in text


def test_send_raises_on_api_failure():
    notifier = _notifier()
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(ok=False, text="Unauthorized")
        with pytest.raises(RuntimeError, match="Telegram"):
            notifier.send(_entry_event(), price=212.40, stop=205.10, shares=47)

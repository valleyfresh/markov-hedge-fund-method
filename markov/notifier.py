from __future__ import annotations

import requests

from markov.signal_router import SignalEvent

_EMOJI = {
    "LONG_ENTRY":  "🟢",
    "SHORT_ENTRY": "🔴",
    "LONG_EXIT":   "⬜",
    "SHORT_EXIT":  "⬜",
}


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str) -> None:
        self._url = f"https://api.telegram.org/bot{token}/sendMessage"
        self._chat_id = chat_id

    def send(self, event: SignalEvent, price: float, stop: float, shares: float) -> None:
        text = self._format(event, price, stop, shares)
        resp = requests.post(self._url, json={"chat_id": self._chat_id, "text": text})
        if not resp.ok:
            raise RuntimeError(f"Telegram send failed: {resp.text}")

    def _format(self, event: SignalEvent, price: float, stop: float, shares: float) -> str:
        emoji = _EMOJI[event.action]
        risk_pct_str = f"{int(event.risk_pct * 100)}%"

        if event.action in ("LONG_ENTRY", "SHORT_ENTRY"):
            direction = "LONG" if event.action == "LONG_ENTRY" else "SHORT"
            return (
                f"{emoji} {direction} signal — {event.ticker}\n"
                f"   Conviction: {event.conviction:.2f} ({event.conviction_tier}) → {risk_pct_str} risk\n"
                f"   Daily regime: {event.ticker_regime.name}\n"
                f"   Close price at signal: ~${price:.2f}  (use 1H chart for actual entry)\n"
                f"   ATR stop: ${stop:.2f}  (1.5×ATR14)\n"
                f"   Position size: {shares:.0f} shares\n"
                f"   SPY: {event.spy_regime.name} — gate open\n"
                f"   → Drop to 1H for entry timing"
            )

        direction = "LONG" if event.action == "LONG_EXIT" else "SHORT"
        trigger = (
            f"regime flipped to {event.ticker_regime.name}"
            if event.action == "LONG_EXIT" and event.ticker_regime.name == "Bear"
            else f"SPY turned {event.spy_regime.name}"
        )
        return (
            f"{emoji} EXIT {direction} — {event.ticker}\n"
            f"   Trigger: {trigger}\n"
            f"   Exit price: ~${price:.2f}"
        )

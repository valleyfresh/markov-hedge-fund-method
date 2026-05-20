# Phase 3 — Live Signal Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a daily pipeline that fetches prices, computes regimes, applies the SPY gate, and fires Telegram alerts for new LONG/SHORT entries and exits. Runs on Ubuntu via cron at 4:30 PM ET on weekdays.

**Architecture:** `SignalRouter` compares today's regime to the stored position in `state/positions.json` and emits `SignalEvent` objects. `TelegramNotifier` formats and sends them. `pipeline.py` is the daily orchestrator that ties all three together. `RegimeEngine` from Phase 1 is the only shared dependency.

**Tech Stack:** Python 3.10+, numpy, pandas, yfinance, requests, pytest, PyYAML, unittest.mock

**Prerequisite:** Phase 1 complete. Phase 2 is independent (can be run in any order after Phase 1).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `markov/signal_router.py` | `SignalEvent`, `SignalRouter` (SPY gate + transition detection) |
| Create | `markov/notifier.py` | `TelegramNotifier` |
| Create | `markov/pipeline.py` | Daily orchestrator entry point |
| Create | `state/positions.json` | Persisted position state (created by pipeline on first run) |
| Create | `tests/test_signal_router.py` | Full test suite for SignalRouter |
| Create | `tests/test_notifier.py` | Test suite for TelegramNotifier (mocked HTTP) |
| Create | `logs/.gitkeep` | Ensure logs dir is tracked |

---

## Task 1: SignalEvent + SignalRouter (TDD)

**Files:**
- Create: `markov/signal_router.py`
- Create: `tests/test_signal_router.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_signal_router.py
import pytest
from markov.engine import Regime
from markov.signal_router import SignalEvent, SignalRouter


def _router() -> SignalRouter:
    return SignalRouter()


# ── Entry signal tests ───────────────────────────────────────────────────────

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


def test_no_entry_when_already_long():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bull, Regime.Bull, "long", 0.80)
    assert event is None


def test_no_entry_when_already_short():
    router = _router()
    event = router.evaluate("AAPL", Regime.Bear, Regime.Bear, "short", 0.80)
    assert event is None


# ── Exit signal tests ────────────────────────────────────────────────────────

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


# ── SignalEvent field tests ──────────────────────────────────────────────────

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
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_signal_router.py -v
```

Expected: FAILED — `markov.signal_router` not found.

- [ ] **Step 3: Create markov/signal_router.py**

```python
# markov/signal_router.py
from __future__ import annotations

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
    (0.70, 1.00, "HIGH",   0.03),
]


def _tier(conviction: float) -> tuple[str, float]:
    for low, high, label, pct in _TIERS:
        if low <= conviction < high:
            return label, pct
    return "HIGH", 0.03


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

        # flat — check entry
        if ticker_regime == Regime.Bull and spy_regime != Regime.Bear:
            return _event("LONG_ENTRY")
        if ticker_regime == Regime.Bear and spy_regime != Regime.Bull:
            return _event("SHORT_ENTRY")
        return None
```

- [ ] **Step 4: Run all signal router tests**

```bash
pytest tests/test_signal_router.py -v
```

Expected: 18 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add markov/signal_router.py tests/test_signal_router.py
git commit -m "feat: add SignalRouter with SPY gate + conviction tiers (TDD)"
```

---

## Task 2: TelegramNotifier (TDD with mocked HTTP)

**Files:**
- Create: `markov/notifier.py`
- Create: `tests/test_notifier.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_notifier.py
from unittest.mock import MagicMock, patch
import pandas as pd
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
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_notifier.py -v
```

Expected: FAILED — `markov.notifier` not found.

- [ ] **Step 3: Create markov/notifier.py**

```python
# markov/notifier.py
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
```

- [ ] **Step 4: Run all notifier tests**

```bash
pytest tests/test_notifier.py -v
```

Expected: 6 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add markov/notifier.py tests/test_notifier.py
git commit -m "feat: add TelegramNotifier with formatted messages (TDD)"
```

---

## Task 3: positions.json state manager (TDD)

**Files:**
- Create: `markov/state.py`
- Modify: `tests/test_signal_router.py` (add state tests)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_signal_router.py`:

```python
from markov.state import PositionState


def test_position_state_defaults_to_flat(tmp_path):
    ps = PositionState(tmp_path / "positions.json")
    assert ps.side("AAPL") == "flat"


def test_position_state_save_and_load(tmp_path):
    path = tmp_path / "positions.json"
    ps = PositionState(path)
    ps.open("AAPL", side="long", entry_price=212.40, stop_price=205.10, conviction=0.74, risk_pct=0.03)
    ps.save()

    ps2 = PositionState(path)
    assert ps2.side("AAPL") == "long"
    assert ps2.entry_price("AAPL") == pytest.approx(212.40)


def test_position_state_close(tmp_path):
    ps = PositionState(tmp_path / "positions.json")
    ps.open("AAPL", side="long", entry_price=212.40, stop_price=205.10, conviction=0.74, risk_pct=0.03)
    ps.close("AAPL")
    assert ps.side("AAPL") == "flat"
```

- [ ] **Step 2: Run to verify new tests fail**

```bash
pytest tests/test_signal_router.py -v
```

Expected: 18 PASSED, 3 FAILED.

- [ ] **Step 3: Create markov/state.py**

```python
# markov/state.py
from __future__ import annotations

import json
from pathlib import Path


class PositionState:
    """Persists open position metadata across daily pipeline runs."""

    def __init__(self, path: str | Path = "state/positions.json") -> None:
        self._path = Path(path)
        self._data: dict[str, dict] = {}
        if self._path.exists():
            self._data = json.loads(self._path.read_text())

    def side(self, ticker: str) -> str:
        return self._data.get(ticker, {}).get("side", "flat")

    def entry_price(self, ticker: str) -> float:
        return float(self._data.get(ticker, {}).get("entry_price", 0.0))

    def stop_price(self, ticker: str) -> float:
        return float(self._data.get(ticker, {}).get("stop_price", 0.0))

    def open(
        self,
        ticker: str,
        *,
        side: str,
        entry_price: float,
        stop_price: float,
        conviction: float,
        risk_pct: float,
    ) -> None:
        self._data[ticker] = {
            "side": side,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "conviction": conviction,
            "risk_pct": risk_pct,
        }

    def risk_pct(self, ticker: str) -> float:
        return float(self._data.get(ticker, {}).get("risk_pct", 0.01))

    def close(self, ticker: str) -> None:
        self._data[ticker] = {"side": "flat"}

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_signal_router.py -v
```

Expected: 21 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add markov/state.py tests/test_signal_router.py
git commit -m "feat: add PositionState for persisting open positions (TDD)"
```

---

## Task 4: Pipeline orchestrator

**Files:**
- Create: `markov/pipeline.py`
- Create: `state/.gitkeep`
- Create: `logs/.gitkeep`

No unit test for the orchestrator itself — it is a thin composition layer. Verify by dry-running with `--dry-run`.

- [ ] **Step 1: Create the pipeline**

```python
# markov/pipeline.py
"""Daily signal pipeline: python -m markov.pipeline [--dry-run]"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yfinance as yf

from markov.config import load_config
from markov.engine import Regime, RegimeEngine
from markov.notifier import TelegramNotifier
from markov.signal_router import SignalRouter
from markov.state import PositionState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _atr14(ohlcv) -> float:
    high, low, close = ohlcv["High"], ohlcv["Low"], ohlcv["Close"]
    prev = close.shift(1)
    tr = np.c_[
        (high - low).to_numpy(),
        (high - prev).abs().to_numpy(),
        (low  - prev).abs().to_numpy(),
    ].max(axis=1)
    atr = float(tr[-14:].mean()) if len(tr) >= 14 else float(tr.mean())
    return atr if not np.isnan(atr) else float(close.iloc[-1]) * 0.02


def main(dry_run: bool = False) -> None:
    cfg = load_config("config.yaml")
    watchlist  = cfg["watchlist"]
    spy_ticker = cfg["spy_ticker"]
    atr_mult   = cfg["risk"]["atr_multiplier"]
    portfolio  = cfg["risk"]["portfolio_value"]

    log.info("Fetching prices for %s + SPY", watchlist)
    all_tickers = list({spy_ticker} | set(watchlist))
    import pandas as pd
    raw = yf.download(all_tickers, period="60d", auto_adjust=True, progress=False)
    is_multi = isinstance(raw.columns, pd.MultiIndex)

    def _ohlcv(ticker: str):
        return raw.xs(ticker, axis=1, level=1) if is_multi else raw

    spy_ohlcv  = _ohlcv(spy_ticker)
    spy_engine = RegimeEngine(cfg["lookback_days"], cfg["threshold_pct"]).fit(spy_ohlcv["Close"])
    spy_regime = spy_engine.current_regime()
    log.info("SPY regime: %s", spy_regime.name)

    state    = PositionState("state/positions.json")
    router   = SignalRouter()
    notifier = TelegramNotifier(
        token=cfg["telegram"]["token"],
        chat_id=cfg["telegram"]["chat_id"],
    ) if not dry_run else None

    for ticker in watchlist:
        try:
            ohlcv = _ohlcv(ticker)
        except KeyError:
            log.warning("No data for %s — skipping", ticker)
            continue

        ticker_engine = RegimeEngine(cfg["lookback_days"], cfg["threshold_pct"]).fit(ohlcv["Close"])
        ticker_regime = ticker_engine.current_regime()
        _, conviction = ticker_engine.signal()
        current_side  = state.side(ticker)

        event = router.evaluate(ticker, ticker_regime, spy_regime, current_side, conviction)

        if event is None:
            log.info("%s: HOLD (%s, side=%s)", ticker, ticker_regime.name, current_side)
            continue

        price = float(ohlcv["Close"].iloc[-1])
        atr   = _atr14(ohlcv)

        if event.action == "LONG_ENTRY":
            stop   = price - atr_mult * atr
            shares = (portfolio * event.risk_pct) / max(abs(price - stop), 0.01)
            state.open(ticker, side="long", entry_price=price, stop_price=stop,
                       conviction=conviction, risk_pct=event.risk_pct)
        elif event.action == "SHORT_ENTRY":
            stop   = price + atr_mult * atr
            shares = (portfolio * event.risk_pct) / max(abs(price - stop), 0.01)
            state.open(ticker, side="short", entry_price=price, stop_price=stop,
                       conviction=conviction, risk_pct=event.risk_pct)
        else:
            stop   = state.stop_price(ticker)
            shares = (portfolio * state.risk_pct(ticker)) / max(abs(state.entry_price(ticker) - stop), 0.01)
            state.close(ticker)

        log.info("%s: %s (conviction=%.2f, price=%.2f, stop=%.2f)",
                 ticker, event.action, conviction, price, stop)

        if dry_run:
            log.info("[DRY RUN] Would send Telegram: %s %s", event.action, ticker)
        else:
            notifier.send(event, price=price, stop=stop, shares=shares)

    state.save()
    log.info("Pipeline complete. State saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Skip Telegram sends")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
```

- [ ] **Step 2: Create state and logs placeholders**

```bash
mkdir -p state logs
touch state/.gitkeep logs/.gitkeep
```

- [ ] **Step 3: Dry-run to verify the pipeline works end-to-end**

```bash
python -m markov.pipeline --dry-run
```

Expected output (no Telegram sends):
```
2026-05-20 16:30:01 INFO Fetching prices for ['AAPL', 'NVDA', ...] + SPY
2026-05-20 16:30:04 INFO SPY regime: Bull
2026-05-20 16:30:04 INFO AAPL: HOLD (Bull, side=flat)
...
2026-05-20 16:30:04 INFO Pipeline complete. State saved.
```

If you see a `LONG_ENTRY` or `SHORT_ENTRY` event, it will print `[DRY RUN] Would send Telegram: ...` — that is correct.

- [ ] **Step 4: Commit**

```bash
git add markov/pipeline.py state/.gitkeep logs/.gitkeep
git commit -m "feat: add daily pipeline orchestrator with --dry-run mode"
```

---

## Task 5: Cron setup (Ubuntu)

- [ ] **Step 1: Confirm repo path**

```bash
pwd
```

Note the full path — you will need it in the cron entry.

- [ ] **Step 2: Set Telegram credentials as environment variables**

Add to `~/.bashrc` (or `~/.zshrc`):

```bash
export TELEGRAM_TOKEN="your-bot-token-here"
export TELEGRAM_CHAT_ID="your-chat-id-here"
```

Then reload:

```bash
source ~/.bashrc
```

To get your bot token: message @BotFather on Telegram → `/newbot`. To get your chat ID: message @userinfobot.

- [ ] **Step 3: Test that credentials are loaded correctly**

```bash
python -c "from markov.config import load_config; c = load_config(); print(c['telegram'])"
```

Expected: `{'token': 'your-bot-token-here', 'chat_id': 'your-chat-id-here'}` (not the `${...}` placeholders).

- [ ] **Step 4: Add cron entry**

```bash
crontab -e
```

Add this line (replace `/home/dlee1/repo/markov-hedge-fund-method` with the actual path from Step 1):

```
30 21 * * 1-5 cd /home/dlee1/repo/markov-hedge-fund-method && TELEGRAM_TOKEN=$TELEGRAM_TOKEN TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID /usr/bin/python3 -m markov.pipeline >> logs/pipeline.log 2>&1
```

The time is `21:30 UTC` = `16:30 ET` (US Eastern, no DST adjustment — adjust to `20:30 UTC` during EDT).

- [ ] **Step 5: Verify cron will fire correctly**

```bash
crontab -l
```

Expected: the line from Step 4 appears in the output.

- [ ] **Step 6: Run the live pipeline once manually to confirm Telegram works**

```bash
python -m markov.pipeline
```

Check your Telegram chat. If you have open positions in `state/positions.json` from a prior dry run, you may see exit signals — that is expected.

- [ ] **Step 7: Commit final state**

```bash
git add markov/ tests/ state/.gitkeep logs/.gitkeep
git commit -m "feat: complete Phase 3 — live pipeline with Telegram alerts and cron"
```

---

## Phase 3 complete

The full system is now operational:

| Component | Status |
|-----------|--------|
| `markov/engine.py` | Canonical regime model, all bugs fixed |
| `config.yaml` | Single source of truth |
| `pine-script/` | Aligned encoding, simple return |
| `markov/backtest.py` | Walk-forward with costs, SPY gate, conviction sizing |
| `markov/signal_router.py` | SPY gate + long/short entry/exit detection |
| `markov/notifier.py` | Formatted Telegram alerts |
| `markov/pipeline.py` | Daily orchestrator, cron-ready |
| `state/positions.json` | Persisted position tracking |

Run the full test suite to confirm everything is green:

```bash
pytest tests/ -v
```

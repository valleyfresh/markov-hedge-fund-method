# Markov Hedge Fund Method — Production Design

**Date:** 2026-05-20
**Framework by:** Roan (@RohOnChain)
**Status:** Approved, pending implementation

---

## Overview

Three-phase production upgrade of the Markov regime signal:

| Phase | Deliverable | Bugs fixed |
|-------|-------------|------------|
| 1 | `markov/` package, RegimeEngine, config.yaml, Pine Script fixes | All 4 |
| 2 | WalkForwardBacktester, results CSV | — |
| 3 | Daily pipeline, SignalRouter, TelegramNotifier, cron setup | — |

Each phase ships independently. The system covers both long and short signals.

---

## Architecture & Data Flow

```
DAILY PIPELINE  (runs ~30 min after market close on Ubuntu)

  yfinance.download(SPY + watchlist, period="60d")
        │
        ├──► SPY RegimeEngine  ──► spy_regime ∈ {Bull, Bear, Sideways}
        │                                ↓
        └──► Ticker RegimeEngine×N      SPY gate
                     │                    │
                     ▼                    ▼
              ticker_signal      AND logic (see below)
                     │
                     ▼
              SignalRouter (per ticker, reads/writes positions.json)
              ├── LONG ENTRY?   → TelegramNotifier
              ├── SHORT ENTRY?  → TelegramNotifier
              ├── LONG EXIT?    → TelegramNotifier
              ├── SHORT EXIT?   → TelegramNotifier
              └── HOLD          → silent

TradingView (live, separate)
  Fixed Pine indicator on Daily chart → regime label + signal overlay
  Drop to 1H chart for manual entry timing after Telegram fires
```

---

## Signal Logic (SPY Gate)

| SPY regime | LONG entry | SHORT entry |
|------------|-----------|-------------|
| Bull       | ✓ ticker → Bull | ✗ suppressed |
| Sideways   | ✓ ticker → Bull | ✓ ticker → Bear |
| Bear       | ✗ suppressed | ✓ ticker → Bear |

**Exit conditions:**
- Long exit: ticker → Bear **or** SPY → Bear (whichever comes first)
- Short exit: ticker → Bull **or** SPY → Bull (whichever comes first)

Signals fire only on regime *transitions*, not on held state. Silent days produce no Telegram messages.

---

## Timeframe

- **Regime computation:** Daily bars (Python pipeline + Pine indicator)
- **Regime lookback:** 20 trading days (≈ 4 calendar weeks)
- **Entry timing:** 1H chart (manual, TradingView) — Telegram alert prompts this
- **ATR for stops:** Daily ATR(14) — sized to regime, not entry bar

---

## Module Structure

```
markov-hedge-fund-method/
├── markov/
│   ├── __init__.py
│   ├── engine.py           ← RegimeEngine class
│   ├── backtest.py         ← WalkForwardBacktester
│   ├── signal_router.py    ← SignalRouter (SPY gate + position state)
│   ├── notifier.py         ← TelegramNotifier
│   └── pipeline.py         ← daily orchestrator (entry point)
├── config.yaml             ← single source of truth for all parameters
├── state/
│   └── positions.json      ← persists open positions across daily runs
├── results/                ← backtest output CSVs
├── logs/                   ← pipeline run logs
├── scripts/                ← original files kept for reference, not imported
├── pine-script/            ← fixed .pine files
└── docs/superpowers/specs/ ← this file
```

---

## RegimeEngine (engine.py)

Canonical regime model used identically by backtest and live pipeline.

```
Encoding: 0 = Bull, 1 = Bear, 2 = Sideways  (unambiguous, used everywhere)

Public API:
  .fit(prices: pd.Series)       → builds transition matrix on history
  .current_regime() → Regime    → returns Regime enum {Bull, Bear, Sideways}
  .signal() → (Regime, float)   → (direction, conviction_score 0–1)
  .stationary_dist() → dict     → π vector, correctly labeled
  .n_step_forecast(n) → dict    → Chapman-Kolmogorov M^n probabilities

conviction_score = |bull_prob − bear_prob|
```

**Bugs fixed in RegimeEngine:**
- Bug 1 (Pine label flip): encoding standardised, Pine rewritten to match
- Bug 2 (log vs simple return): uses `pct_change` throughout; Pine updated to `(close - close[n]) / close[n]`
- Bug 3 (`if i == i`): removed in pipeline rewrite
- Bug 4 (2% vs 5% threshold): `config.yaml` is sole source, default 5%

---

## WalkForwardBacktester (backtest.py)

```
Train window:  252 trading days (rolling)
Test window:   21 trading days  (1 month — matches expected swing hold)
Step:          21 trading days  (non-overlapping test windows)

Per test window, per ticker:
  1. Fit RegimeEngine on train window
  2. Apply Daily regime labels to test window (no future data)
  3. Apply SPY gate (same train/test split on SPY)
  4. Generate LONG / SHORT / FLAT signals with position sizing
  5. Compute P&L with costs

Costs applied:
  tx_cost:           0.10% per side
  slippage:          0.05% per side
  short borrow:      0.01% per day held

Output per ticker + aggregate:
  - Equity curve
  - Sharpe ratio (annualised)
  - Max drawdown
  - Win rate, avg win, avg loss
  - Trade log (entry date, exit date, side, conviction, return)
  - Benchmark: SPY buy-and-hold over same period

Results written to: results/backtest_YYYY-MM-DD.csv
```

---

## Risk Management

**Conviction-scaled position sizing:**

```
conviction = |bull_prob − bear_prob|   # 0.0 → 1.0

Tier        Conviction range    Portfolio risk
Low         0.00 – 0.40         1%
Medium      0.40 – 0.70         2%
High        0.70 – 1.00         3%

Position size = (portfolio_value × risk_pct) / (ATR(14) × atr_multiplier)
```

**ATR stop placement:**
```
Long:   stop = entry_price − (ATR(14) × atr_multiplier)
Short:  stop = entry_price + (ATR(14) × atr_multiplier)
```

Stop level computed at signal time, stored in `positions.json`, monitored on each daily run.

---

## Live Pipeline & Telegram Notifier (pipeline.py, notifier.py)

**Pipeline steps (daily):**
1. Load `config.yaml`
2. Fetch prices via yfinance
3. Compute regimes (SPY + watchlist)
4. Load `state/positions.json`
5. Run SignalRouter per ticker
6. Send Telegram alerts for any SignalEvents
7. Save updated `state/positions.json`

**positions.json schema:**
```json
{
  "AAPL": {
    "side": "long",
    "entry_date": "2026-05-15",
    "entry_price": 212.40,
    "stop_price": 205.10,
    "entry_regime": "Bull",
    "conviction": 0.74,
    "risk_pct": 0.03
  },
  "MSFT": {"side": "flat", "entry_date": null}
}
```

**Telegram message format:**
```
🟢 LONG signal — AAPL
   Conviction: 0.74 (HIGH) → 3% risk
   Daily regime: Bull
   Close price at signal: ~$212.40  (use 1H chart for actual entry)
   ATR stop: $205.10  (1.5×ATR14)
   Position size: 47 shares  ($10k account)
   SPY: Sideways — gate open
   → Drop to 1H for entry timing

⬜ EXIT LONG — AAPL
   Trigger: regime flipped Bull → Bear
   Held: 5 days
```

**Cron entry (Ubuntu, weekdays 4:30 PM ET):**
```
30 16 * * 1-5 cd /path/to/repo && python -m markov.pipeline >> logs/pipeline.log 2>&1
```

---

## config.yaml

Single source of truth for all parameters. Credentials via environment variables only.

```yaml
watchlist: [AAPL, NVDA, MSFT, AMZN, META]
spy_ticker: SPY
lookback_days: 20
threshold_pct: 0.05

risk:
  portfolio_value: 10000
  atr_period: 14
  atr_multiplier: 1.5
  conviction_tiers:
    low:    [0.00, 0.40, 0.01]
    medium: [0.40, 0.70, 0.02]
    high:   [0.70, 1.00, 0.03]

backtest:
  train_days: 252
  test_days: 21
  tx_cost_pct: 0.001
  slippage_pct: 0.0005
  short_borrow_daily_pct: 0.0001

telegram:
  token: "${TELEGRAM_TOKEN}"
  chat_id: "${TELEGRAM_CHAT_ID}"
```

---

## Pine Script Fixes (pine-script/)

Three targeted fixes, no structural rewrite:

1. **Return type (Bug 2):** `(close - close[lookback]) / close[lookback]` replaces log return
2. **Label mapping (Bug 1):** regime encoding 0=Bull/1=Bear/2=Sideways applied consistently to transition matrix display and stationary distribution table
3. **Threshold (Bug 4):** single `input.float(0.05, "Regime Threshold")` input, default 5%

Pine indicator is read-only visual on Daily chart. Telegram is the alert surface — no Pine alerts wired.

---

## Known Conceptual Limitations (not fixed in this build)

1. **Autocorrelated labels:** 20-day rolling regime shares 19 days with adjacent label — transitions are not i.i.d. Markov. Acknowledged; production build does not claim otherwise.
2. **HMM lookahead:** HMM fitted on all data affects display only, not the backtest signal.
3. **Momentum proxy:** Regime labeling is structurally a momentum signal. The Markov framing adds transition probability structure on top.

These are deferred to future research phases.

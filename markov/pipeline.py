"""Daily signal pipeline: python -m markov.pipeline [--dry-run]"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from markov.config import load_config
from markov.engine import RegimeEngine
from markov.notifier import TelegramNotifier
from markov.signal_router import SignalRouter
from markov.state import PositionState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _atr14(ohlcv: pd.DataFrame) -> float:
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
    raw = yf.download(all_tickers, period="60d", auto_adjust=True, progress=False)
    is_multi = isinstance(raw.columns, pd.MultiIndex)

    def _ohlcv(ticker: str) -> pd.DataFrame:
        return raw.xs(ticker, axis=1, level=1) if is_multi else raw

    spy_ohlcv  = _ohlcv(spy_ticker)
    spy_engine = RegimeEngine(cfg["lookback_days"], cfg["threshold_pct"]).fit(spy_ohlcv["Close"])
    spy_regime = spy_engine.current_regime()
    log.info("SPY regime: %s", spy_regime.name)

    state    = PositionState("state/positions.json")
    router   = SignalRouter()
    notifier = (
        TelegramNotifier(token=cfg["telegram"]["token"], chat_id=cfg["telegram"]["chat_id"])
        if not dry_run else None
    )

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

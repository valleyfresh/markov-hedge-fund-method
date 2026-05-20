"""Run walk-forward backtest: python -m markov.run_backtest"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from markov.backtest import WalkForwardBacktester
from markov.config import load_config


def main() -> None:
    cfg = load_config("config.yaml")
    watchlist  = cfg["watchlist"]
    spy_ticker = cfg["spy_ticker"]

    all_tickers = list({spy_ticker} | set(watchlist))
    print(f"Fetching {len(all_tickers)} tickers from yfinance...")
    raw = yf.download(all_tickers, period="5y", auto_adjust=True, progress=False)

    is_multi = isinstance(raw.columns, pd.MultiIndex)
    spy_ohlcv = raw.xs(spy_ticker, axis=1, level=1) if is_multi else raw
    ticker_ohlcv = {
        t: raw.xs(t, axis=1, level=1)
        for t in watchlist
        if is_multi and t in raw.columns.get_level_values(1)
    }

    bt = WalkForwardBacktester(cfg)
    results = bt.run(ticker_ohlcv, spy_ohlcv)

    Path("results").mkdir(exist_ok=True)
    out_path = Path(f"results/backtest_{date.today()}.csv")

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "sharpe", "max_drawdown", "win_rate", "avg_win", "avg_loss", "n_trades"])
        for ticker, r in results.items():
            writer.writerow([
                ticker,
                f"{r.sharpe:.3f}",
                f"{r.max_drawdown:.3f}",
                f"{r.win_rate:.3f}",
                f"{r.avg_win:.3f}",
                f"{r.avg_loss:.3f}",
                len(r.trades),
            ])

    print(f"\nResults written to {out_path}\n")
    for ticker, r in results.items():
        print(f"  {ticker:6s}  Sharpe={r.sharpe:+.2f}  MaxDD={r.max_drawdown:.1%}  WinRate={r.win_rate:.0%}  Trades={len(r.trades)}")


if __name__ == "__main__":
    main()

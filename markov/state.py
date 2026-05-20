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

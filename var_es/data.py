"""Données de marché : téléchargement avec cache CSV, rendements, données fictives."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .models import simulate_returns


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def download_prices(tickers: list[str], start: str, end: str,
                    cache_dir: str | Path = "data", force: bool = False) -> pd.DataFrame:
    """Prix ajustés (dividendes et splits) avec dates FIXES et cache CSV.

    Dates fixes + cache = résultats reproductibles (contrairement à period="5y").
    Les colonnes sont renvoyées dans l'ordre de `tickers`.
    """
    tickers = list(tickers)
    name = "_".join(sorted(tickers)) + f"_{start}_{end}.csv"
    path = Path(cache_dir) / f"prices_{name}"
    if path.exists() and not force:
        return load_prices_csv(path)

    import yfinance as yf  # import local : inutile pour les tests hors ligne

    raw = yf.download(tickers, start=start, end=end, interval="1d",
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    prices = raw[tickers].dropna()
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendements simples journaliers."""
    return prices.pct_change().dropna()


def synthetic_returns(tickers: list[str], n_days: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Rendements FICTIFS (Student + régimes de volatilité) pour tester sans connexion.

    À n'utiliser que pour développer et tester le code, jamais pour tirer des conclusions.
    """
    rng = np.random.default_rng(seed)
    k = len(tickers)
    vol = np.linspace(0.010, 0.016, k)
    corr = np.full((k, k), 0.7) + 0.3 * np.eye(k)
    cov = np.outer(vol, vol) * corr
    base = simulate_returns(np.full(k, 0.0004), cov, n_days, rng=rng, dof=5)
    # volatilité stochastique persistante (AR(1) sur le log de la vol)
    h = np.zeros(n_days)
    for t in range(1, n_days):
        h[t] = 0.98 * h[t - 1] + 0.12 * rng.standard_normal()
    returns = base * np.exp(h)[:, None]
    index = pd.bdate_range("2010-01-04", periods=n_days)
    return pd.DataFrame(returns, index=index, columns=tickers)

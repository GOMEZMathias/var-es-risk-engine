"""Comparaison des six modèles de VaR sur les mêmes dates.

Cette fonction est la source unique du tableau de résultats : elle est appelée par
`scripts/run_backtest.py` et par le notebook (section 5.7).
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from .backtest import backtest_summary, rolling_var_es, violations
from .ewma import ewma_var_es


def model_configs(returns: pd.Series, first_forecast: int = 500, window: int = 250,
                  lam: float = 0.94) -> dict[str, Callable[[float], pd.DataFrame]]:
    """Les six modèles comparés : {nom: fonction alpha -> DataFrame(var, es)}."""
    first = first_forecast
    return {
        "gaussienne (croissante)":
            lambda a: rolling_var_es(returns, a, method="gaussian", window=None, start=first),
        f"gaussienne (glissante {window})":
            lambda a: rolling_var_es(returns, a, method="gaussian", window=window, start=first),
        "historique (croissante)":
            lambda a: rolling_var_es(returns, a, method="historical", window=None, start=first),
        f"historique (glissante {window})":
            lambda a: rolling_var_es(returns, a, method="historical", window=window, start=first),
        f"EWMA gaussienne (λ={lam})":
            lambda a: ewma_var_es(returns, a, lam=lam, method="gaussian", start=first),
        f"EWMA + FHS (λ={lam})":
            lambda a: ewma_var_es(returns, a, lam=lam, method="fhs", start=first),
    }


def compare_models(returns: pd.Series, alphas=(0.95, 0.99), first_forecast: int = 500,
                   window: int = 250, lam: float = 0.94):
    """Backtest des six modèles aux niveaux `alphas`, sur les mêmes dates.

    Retourne (tableau, VaR prévues, violations) ; les deux dictionnaires {nom: Series}
    correspondent au premier niveau de `alphas` (utile pour les graphiques).
    """
    alphas = list(alphas)
    configs = model_configs(returns, first_forecast, window, lam)
    rows, plot_var, plot_viol = [], {}, {}
    for alpha in alphas:
        for name, forecast in configs.items():
            f = forecast(alpha)
            v = violations(returns, f["var"])
            rows.append({"alpha": alpha, "modèle": name, **backtest_summary(v, alpha)})
            if alpha == alphas[0]:
                plot_var[name], plot_viol[name] = f["var"], v
    return pd.DataFrame(rows), plot_var, plot_viol

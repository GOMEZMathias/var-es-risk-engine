"""Volatilité dynamique EWMA (RiskMetrics) et VaR/ES associées.

Variance conditionnelle prévue pour le jour t, à partir des rendements jusqu'à t-1 :

    sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r_{t-1}^2

La moyenne des rendements est supposée nulle (convention RiskMetrics, raisonnable à 1 jour).
Deux façons d'en déduire une VaR :
  - "gaussian" : VaR = sigma_t * z_alpha, ES = sigma_t * phi(z_alpha) / (1 - alpha)
  - "fhs"      : Filtered Historical Simulation. On standardise les pertes passées par la
                 volatilité prévue ce jour-là, on prend le quantile empirique de ces pertes
                 standardisées (queues réelles, non gaussiennes), puis on le remultiplie par
                 la volatilité prévue pour demain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .models import _check_alpha


def ewma_volatility(returns: pd.Series, lam: float = 0.94, warmup: int = 30) -> pd.Series:
    """Volatilité journalière PRÉVUE pour chaque jour, à partir du passé uniquement.

    La première prévision (jour d'indice `warmup`) est initialisée avec l'écart-type des
    `warmup` premiers rendements ; ensuite la récursion EWMA s'applique.
    Renvoie une Series indexée par date, à partir du jour d'indice `warmup`.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lam doit être dans (0, 1)")
    if warmup < 2:
        raise ValueError("warmup doit être >= 2")
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n <= warmup:
        raise ValueError("pas assez de données pour le warmup")

    sigma2 = np.empty(n - warmup)
    sigma2[0] = r[:warmup].var(ddof=1)
    for k in range(1, n - warmup):
        t = warmup + k
        sigma2[k] = lam * sigma2[k - 1] + (1.0 - lam) * r[t - 1] ** 2
    return pd.Series(np.sqrt(sigma2), index=returns.index[warmup:], name="sigma")


def ewma_var_es(returns: pd.Series, alpha: float, lam: float = 0.94, method: str = "gaussian",
                start: int = 500, warmup: int = 30) -> pd.DataFrame:
    """VaR et ES prévues avec une volatilité EWMA (en fraction du portefeuille).

    method : "gaussian" ou "fhs" (voir l'en-tête du module).
    start  : position de la première prévision retournée (>= warmup).
    Renvoie un DataFrame indexé par date avec les colonnes "var" et "es".
    """
    _check_alpha(alpha)
    if method not in ("gaussian", "fhs"):
        raise ValueError("method doit être 'gaussian' ou 'fhs'")
    if start < warmup:
        raise ValueError("start doit être >= warmup")

    sigma = ewma_volatility(returns, lam=lam, warmup=warmup)  # indexée à partir de `warmup`
    sig = sigma.to_numpy()
    losses = -np.asarray(returns, dtype=float)[warmup:]       # alignées sur sigma
    first = start - warmup
    n = len(sig)

    if method == "gaussian":
        z = norm.ppf(alpha)
        var = sig[first:] * z
        es = sig[first:] * norm.pdf(z) / (1.0 - alpha)
    else:
        std_losses = losses / sig          # pertes standardisées par la volatilité prévue
        var, es = np.empty(n - first), np.empty(n - first)
        for k, i in enumerate(range(first, n)):
            past = std_losses[:i]          # jours strictement antérieurs à i
            q = np.quantile(past, alpha)
            var[k] = sig[i] * q
            es[k] = sig[i] * past[past >= q].mean()
    return pd.DataFrame({"var": var, "es": es}, index=sigma.index[first:])

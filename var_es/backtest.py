"""Backtesting de la VaR : prévisions hors échantillon, violations, tests statistiques.

Toutes les prévisions de VaR pour le jour t n'utilisent que les rendements STRICTEMENT
antérieurs à t (pas de look-ahead). Les VaR sont exprimées en fraction du portefeuille
(multiplier par la valeur initiale pour obtenir des euros).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, norm

from .models import _check_alpha


# --------------------------------------------------------------------------
# Prévisions glissantes
# --------------------------------------------------------------------------
def rolling_var_es(returns: pd.Series, alpha: float, method: str = "historical",
                   window: int | None = None, start: int | None = None) -> pd.DataFrame:
    """VaR et ES prévues pour chaque jour, à partir du passé uniquement.

    method : "historical" (quantile empirique) ou "gaussian" (moyenne/écart-type).
    window : taille de la fenêtre glissante ; None = fenêtre croissante (expanding).
    start  : position (en nombre d'observations) de la première prévision.
             Par défaut `window` si défini, sinon 250.
    Renvoie un DataFrame indexé par date avec les colonnes "var" et "es".
    """
    _check_alpha(alpha)
    if method not in ("historical", "gaussian"):
        raise ValueError("method doit être 'historical' ou 'gaussian'")
    first = start if start is not None else (window if window is not None else 250)
    if window is not None and first < window:
        raise ValueError("start doit être >= window pour disposer d'une fenêtre complète")

    losses = -np.asarray(returns, dtype=float)
    n = len(losses)
    var_out, es_out = np.empty(n - first), np.empty(n - first)
    z = norm.ppf(alpha)
    es_factor = norm.pdf(z) / (1.0 - alpha)

    for k, i in enumerate(range(first, n)):
        hist = losses[(i - window if window is not None else 0):i]  # exclut le jour i
        if method == "historical":
            v = np.quantile(hist, alpha)
            var_out[k], es_out[k] = v, hist[hist >= v].mean()
        else:
            m, s = hist.mean(), hist.std(ddof=1)
            var_out[k], es_out[k] = m + s * z, m + s * es_factor
    return pd.DataFrame({"var": var_out, "es": es_out}, index=returns.index[first:])


def violations(returns: pd.Series, var_series: pd.Series) -> pd.Series:
    """Série booléenne : la perte réalisée dépasse-t-elle la VaR prévue ?"""
    realized_loss = -returns.loc[var_series.index]
    return (realized_loss > var_series).rename("violation")


# --------------------------------------------------------------------------
# Tests statistiques
# --------------------------------------------------------------------------
def kupiec_pof(viol, alpha: float) -> tuple[float, float]:
    """Test de Kupiec (couverture inconditionnelle). Renvoie (LR_uc, p-value).

    H0 : la fréquence de violations vaut p = 1 - alpha. LR ~ chi2(1) sous H0.
    """
    _check_alpha(alpha)
    v = np.asarray(viol, dtype=int)
    n, x = len(v), int(v.sum())
    p, p_hat = 1.0 - alpha, x / n
    lr = 2.0 * (xlogy(x, p_hat / p) + xlogy(n - x, (1.0 - p_hat) / (1.0 - p)))
    return float(lr), float(chi2.sf(lr, df=1))


def count_transitions(viol) -> tuple[int, int, int, int]:
    """Compte (n00, n01, n10, n11) : transitions jour t-1 -> jour t."""
    v = np.asarray(viol, dtype=int)
    a, b = v[:-1], v[1:]
    return (int(np.sum((a == 0) & (b == 0))), int(np.sum((a == 0) & (b == 1))),
            int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 1) & (b == 1))))


def christoffersen_independence(viol) -> tuple[float, float]:
    """Test d'indépendance de Christoffersen. Renvoie (LR_ind, p-value).

    H0 : P(violation | veille sans violation) = P(violation | veille avec violation).
    """
    n00, n01, n10, n11 = count_transitions(viol)
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_indep = xlogy(n01 + n11, pi) + xlogy(n00 + n10, 1.0 - pi)
    ll_dep = (xlogy(n01, pi0) + xlogy(n00, 1.0 - pi0)
              + xlogy(n11, pi1) + xlogy(n10, 1.0 - pi1))
    lr = max(0.0, 2.0 * (ll_dep - ll_indep))
    return float(lr), float(chi2.sf(lr, df=1))


def christoffersen_conditional_coverage(viol, alpha: float) -> tuple[float, float]:
    """Couverture conditionnelle : LR_cc = LR_uc + LR_ind ~ chi2(2) sous H0."""
    lr = kupiec_pof(viol, alpha)[0] + christoffersen_independence(viol)[0]
    return float(lr), float(chi2.sf(lr, df=2))


def basel_traffic_light(n_violations: int) -> str:
    """Zone de Bâle pour 250 jours de test à 99 % : vert (0-4), orange (5-9), rouge (10+).

    Valable uniquement pour ce cadre précis (250 observations, VaR 99 %).
    """
    if n_violations <= 4:
        return "vert"
    return "orange" if n_violations <= 9 else "rouge"


def backtest_summary(viol, alpha: float) -> dict:
    """Résumé : violations observées/attendues, Kupiec, indépendance, couverture conditionnelle."""
    v = np.asarray(viol, dtype=int)
    n, x = len(v), int(v.sum())
    lr_uc, p_uc = kupiec_pof(v, alpha)
    lr_ind, p_ind = christoffersen_independence(v)
    lr_cc, p_cc = christoffersen_conditional_coverage(v, alpha)
    return {
        "n_obs": n, "violations": x, "attendues": round(n * (1 - alpha), 1),
        "taux": x / n,
        "LR_kupiec": lr_uc, "p_kupiec": p_uc,
        "LR_ind": lr_ind, "p_ind": p_ind,
        "LR_cc": lr_cc, "p_cc": p_cc,
    }

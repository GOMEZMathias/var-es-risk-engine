"""Construction du portefeuille. Les poids sont TOUJOURS indexés par ticker.

Piège classique : `yf.download` renvoie les colonnes triées par ordre alphabétique,
pas dans l'ordre de la liste demandée. Multiplier par un tableau de poids positionnel
applique alors les poids au mauvais actif. Ici l'alignement se fait par nom de colonne.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_weights(weights: dict[str, float], tol: float = 1e-8) -> pd.Series:
    """Série de poids indexée par ticker ; vérifie que la somme vaut 1."""
    s = pd.Series(weights, dtype=float)
    if abs(s.sum() - 1.0) > tol:
        raise ValueError(f"Les poids doivent sommer à 1, somme = {s.sum():.6f}")
    return s


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Rendement journalier du portefeuille (poids constants, alignés par ticker)."""
    missing = set(weights.index) - set(returns.columns)
    if missing:
        raise KeyError(f"Tickers absents des rendements : {sorted(missing)}")
    aligned = returns[list(weights.index)]
    return aligned.dot(weights).rename("portfolio")


def portfolio_losses(returns: pd.DataFrame, weights: pd.Series, v0: float = 1.0) -> pd.Series:
    """Pertes journalières (positives = perte) pour une valeur initiale v0."""
    return (-v0 * portfolio_returns(returns, weights)).rename("loss")


def covariance_from_vol_corr(vol, corr) -> np.ndarray:
    """Matrice de covariance à partir des volatilités et de la matrice de corrélation."""
    vol = np.asarray(vol, dtype=float)
    return np.outer(vol, vol) * np.asarray(corr, dtype=float)

"""Estimation de la VaR et de l'Expected Shortfall (ES).

Convention : une *perte* est positive. Pour un niveau de confiance alpha dans (0, 1) :

    VaR_alpha(L) = q_alpha(L)
    ES_alpha(L)  = E[L | L >= VaR_alpha(L)]      (loi continue)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha doit être dans (0, 1), reçu {alpha!r}")


# --------------------------------------------------------------------------
# Méthode historique / empirique (marche aussi sur des pertes simulées)
# --------------------------------------------------------------------------
def var_es_empirical(losses, alpha: float) -> tuple[float, float]:
    """VaR et ES empiriques d'un échantillon de pertes (historique ou simulé)."""
    _check_alpha(alpha)
    losses = np.asarray(losses, dtype=float)
    var = np.quantile(losses, alpha)
    es = losses[losses >= var].mean()
    return float(var), float(es)


# --------------------------------------------------------------------------
# Méthode gaussienne (formules fermées)
# --------------------------------------------------------------------------
def var_es_gaussian(mean_loss: float, std_loss: float, alpha: float) -> tuple[float, float]:
    """VaR et ES d'une perte L ~ N(mean_loss, std_loss^2).

    VaR = m + s * z,   ES = m + s * phi(z) / (1 - alpha),   z = Phi^{-1}(alpha)
    """
    _check_alpha(alpha)
    z = norm.ppf(alpha)
    var = mean_loss + std_loss * z
    es = mean_loss + std_loss * norm.pdf(z) / (1.0 - alpha)
    return float(var), float(es)


def gaussian_loss_params(mu, cov, weights, v0: float) -> tuple[float, float]:
    """Moyenne et écart-type (en euros) de la perte d'un portefeuille gaussien.

    `mu`, `cov` et `weights` doivent être alignés dans le même ordre d'actifs.
    """
    mu, cov, w = (np.asarray(x, dtype=float) for x in (mu, cov, weights))
    mean_loss = -v0 * float(w @ mu)
    std_loss = v0 * float(np.sqrt(w @ cov @ w))
    return mean_loss, std_loss


# --------------------------------------------------------------------------
# Monte-Carlo
# --------------------------------------------------------------------------
def simulate_returns(mu, cov, n: int, rng: np.random.Generator | None = None,
                     dof: float | None = None) -> np.ndarray:
    """Simule n vecteurs de rendements.

    dof=None : loi normale multivariée N(mu, cov).
    dof>2    : loi de Student multivariée, normalisée pour que la matrice de
               covariance soit exactement `cov` (queues plus épaisses).
    """
    rng = np.random.default_rng() if rng is None else rng
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if dof is None:
        return rng.multivariate_normal(mu, cov, size=n)
    if dof <= 2:
        raise ValueError("dof doit être > 2 pour que la covariance existe")
    z = rng.multivariate_normal(np.zeros(len(mu)), cov * (dof - 2) / dof, size=n)
    w = rng.chisquare(dof, size=n)
    return mu + z / np.sqrt(w / dof)[:, None]


def var_es_monte_carlo(mu, cov, weights, v0: float, alpha: float, n: int,
                       rng: np.random.Generator | None = None,
                       dof: float | None = None) -> tuple[float, float]:
    """VaR et ES (en euros) estimées par Monte-Carlo sur n scénarios."""
    returns = simulate_returns(mu, cov, n, rng=rng, dof=dof)
    losses = -v0 * (returns @ np.asarray(weights, dtype=float))
    return var_es_empirical(losses, alpha)

"""Étude de convergence Monte-Carlo de la VaR et de l'ES, moyennée sur plusieurs répétitions.

Une seule exécution donne une courbe d'erreur bruitée : on répète l'expérience `n_rep`
fois et on moyenne l'erreur absolue. La référence est la formule gaussienne fermée.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import gaussian_loss_params, var_es_gaussian, var_es_monte_carlo


def mc_convergence(mu, cov, weights, v0: float, alpha: float, sizes,
                   n_rep: int = 100, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Erreur absolue moyenne (en euros) de la VaR/ES Monte-Carlo vs la formule fermée.

    Renvoie un DataFrame : n, var_mae, es_mae (erreur absolue moyenne sur n_rep répétitions).
    """
    rng = np.random.default_rng() if rng is None else rng
    var_ref, es_ref = var_es_gaussian(*gaussian_loss_params(mu, cov, weights, v0), alpha)

    rows = []
    for n in sizes:
        errs = np.array([
            var_es_monte_carlo(mu, cov, weights, v0, alpha, int(n), rng=rng)
            for _ in range(n_rep)
        ])
        rows.append({
            "n": int(n),
            "var_mae": float(np.mean(np.abs(errs[:, 0] - var_ref))),
            "es_mae": float(np.mean(np.abs(errs[:, 1] - es_ref))),
        })
    return pd.DataFrame(rows)

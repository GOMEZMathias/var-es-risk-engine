"""Convergence Monte-Carlo de la VaR/ES.

Usage (depuis la racine du projet) :
    python -m scripts.run_mc_convergence
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import numpy as np

from var_es.convergence import mc_convergence
from var_es.models import gaussian_loss_params, var_es_gaussian
from var_es.plots import plot_convergence, save_fig
from var_es.portfolio import covariance_from_vol_corr

# Hypothèses du notebook (3 actions fictives A, B, C)
V0 = 100_000
WEIGHTS = np.array([0.50, 0.30, 0.20])
MU = np.array([0.0005, 0.0003, 0.0004])
VOL = np.array([0.020, 0.015, 0.025])
CORR = np.array([
    [1.0, 0.6, 0.3],
    [0.6, 1.0, 0.4],
    [0.3, 0.4, 1.0]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--n-rep", type=int, default=100, help="répétitions par taille")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    cov = covariance_from_vol_corr(VOL, CORR)
    var_th, es_th = var_es_gaussian(*gaussian_loss_params(MU, cov, WEIGHTS, V0), args.alpha)
    print(f"Référence théorique à {args.alpha:.0%} : VaR = {var_th:.2f} €, ES = {es_th:.2f} €")

    rng = np.random.default_rng(args.seed)
    df = mc_convergence(MU, cov, WEIGHTS, V0, args.alpha,
                        sizes=[1_000, 10_000, 100_000], n_rep=args.n_rep, rng=rng)
    print(df.round(2).to_string(index=False))

    df.to_csv(f"{args.out}/mc_convergence.csv", index=False)
    save_fig(plot_convergence(df), f"{args.out}/mc_convergence.png")
    print(f"Résultats sauvegardés dans {args.out}/")


if __name__ == "__main__":
    main()

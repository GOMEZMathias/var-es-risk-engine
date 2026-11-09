"""Backtest de la VaR d'un portefeuille SPY/QQQ/IWM (50 % / 30 % / 20 %).

Compare méthode gaussienne et historique, avec fenêtre croissante ou glissante, sur le
PORTEFEUILLE (pas sur chaque ETF). Toutes les configurations sont testées sur les
mêmes dates pour être comparables.

Usage (depuis la racine du projet) :
    python -m scripts.run_backtest                 # données Yahoo Finance (cache dans data/)
    python -m scripts.run_backtest --synthetic     # données fictives, sans connexion
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from var_es.comparison import compare_models
from var_es.data import compute_returns, download_prices, synthetic_returns
from var_es.plots import plot_var_vs_losses, plot_violations, save_fig
from var_es.portfolio import make_weights, portfolio_returns

WEIGHTS = make_weights({"SPY": 0.50, "QQQ": 0.30, "IWM": 0.20})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2026-11-01")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.95, 0.99])
    p.add_argument("--first-forecast", type=int, default=500,
                   help="nombre d'observations avant la première prévision (commun à toutes les configs)")
    p.add_argument("--window", type=int, default=250, help="taille de la fenêtre glissante")
    p.add_argument("--lam", type=float, default=0.94, help="facteur de décroissance EWMA (RiskMetrics : 0,94)")
    p.add_argument("--synthetic", action="store_true", help="données fictives (test hors ligne)")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    tickers = list(WEIGHTS.index)
    if args.synthetic:
        returns = synthetic_returns(tickers)
        print("ATTENTION : données FICTIVES, résultats sans valeur financière.")
    else:
        returns = compute_returns(download_prices(tickers, args.start, args.end))
    port = portfolio_returns(returns, WEIGHTS)
    print(f"{len(port)} rendements journaliers, {port.index[0].date()} -> {port.index[-1].date()}")

    summary, plot_var, plot_viol = compare_models(
        port, args.alphas, first_forecast=args.first_forecast, window=args.window, lam=args.lam)
    cols = ["alpha", "modèle", "n_obs", "violations", "attendues", "taux",
            "p_kupiec", "p_ind", "p_cc"]
    print(summary[cols].round(4).to_string(index=False))

    Path(args.out).mkdir(parents=True, exist_ok=True)
    summary.to_csv(f"{args.out}/backtest_summary.csv", index=False)
    losses = -port.loc[next(iter(plot_var.values())).index]
    save_fig(plot_var_vs_losses(losses, plot_var,
                                title=f"VaR {args.alphas[0]:.0%} prévue vs perte réalisée"),
             f"{args.out}/var_vs_losses.png")
    save_fig(plot_violations(plot_viol), f"{args.out}/violations.png")
    print(f"Résultats sauvegardés dans {args.out}/")


if __name__ == "__main__":
    main()

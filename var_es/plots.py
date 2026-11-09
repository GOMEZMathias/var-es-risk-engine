"""Graphiques. Chaque fonction renvoie la figure matplotlib (à sauvegarder avec save_fig)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import var_es_empirical


def save_fig(fig, path: str | Path, dpi: int = 130) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_loss_distribution(losses, alphas=(0.95, 0.99), title="Distribution des pertes",
                           unit="€", bins=100):
    """Histogramme des pertes avec les VaR (traits) aux niveaux demandés."""
    losses = np.asarray(losses, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(losses, bins=bins, density=True, alpha=0.7)
    for a in alphas:
        var, es = var_es_empirical(losses, a)
        ax.axvline(var, linestyle="--", color="C1" if a == alphas[0] else "C3",
                   label=f"VaR {a:.0%} = {var:,.0f} {unit}")
    ax.set_xlabel(f"Perte ({unit})")
    ax.set_ylabel("Densité")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    return fig


def plot_convergence(df: pd.DataFrame, title="Convergence Monte-Carlo (erreur absolue moyenne)"):
    """Erreur vs nombre de simulations en log-log, avec la pente théorique -1/2."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["n"], df["var_mae"], marker="o", label="VaR")
    ax.plot(df["n"], df["es_mae"], marker="s", label="ES")
    n0, e0 = df["n"].iloc[0], df["var_mae"].iloc[0]
    ax.plot(df["n"], e0 * np.sqrt(n0 / df["n"]), "k--", alpha=0.5, label="pente -1/2 (théorie)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nombre de simulations")
    ax.set_ylabel("Erreur absolue moyenne (€)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return fig


def plot_var_vs_losses(losses: pd.Series, var_dict: dict[str, pd.Series],
                       title="VaR prévue vs perte réalisée"):
    """Perte réalisée (gris) et courbes de VaR prévues pour chaque méthode."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(losses.index, losses.values, color="0.7", lw=0.6, label="Perte réalisée")
    for name, s in var_dict.items():
        ax.plot(s.index, s.values, lw=1.2, label=f"VaR {name}")
    ax.set_title(title)
    ax.set_ylabel("Perte (fraction du portefeuille)")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)
    return fig


def plot_violations(viol_dict: dict[str, pd.Series], title="Chronologie des violations"):
    """Une ligne par méthode : un point à chaque jour de violation (repère les regroupements)."""
    fig, ax = plt.subplots(figsize=(12, 2.0 + 0.6 * len(viol_dict)))
    for k, (name, v) in enumerate(viol_dict.items()):
        days = v.index[v.values.astype(bool)]
        ax.scatter(days, np.full(len(days), k), s=14, label=name)
    ax.set_yticks(range(len(viol_dict)))
    ax.set_yticklabels(list(viol_dict))
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    return fig

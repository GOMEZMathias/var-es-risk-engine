import numpy as np
import pandas as pd

from var_es.backtest import rolling_var_es
from var_es.comparison import compare_models, model_configs


def _returns(n=900, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(rng.normal(0.0003, 0.01, n), index=idx)


def test_six_models_same_dates():
    r = _returns()
    table, plot_var, plot_viol = compare_models(r, alphas=(0.95, 0.99), first_forecast=300)
    assert len(table) == 12                      # 6 modèles x 2 niveaux
    assert table["n_obs"].nunique() == 1         # mêmes dates pour tous
    assert table["n_obs"].iloc[0] == len(r) - 300
    assert len(plot_var) == len(plot_viol) == 6


def test_matches_direct_call():
    r = _returns()
    table, _, _ = compare_models(r, alphas=(0.95,), first_forecast=300)
    f = rolling_var_es(r, 0.95, method="historical", window=None, start=300)
    expected = int((-r.loc[f.index] > f["var"]).sum())
    row = table[table["modèle"] == "historique (croissante)"].iloc[0]
    assert row["violations"] == expected


def test_model_names_follow_parameters():
    names = list(model_configs(_returns(), window=120, lam=0.97))
    assert "historique (glissante 120)" in names and "EWMA + FHS (λ=0.97)" in names

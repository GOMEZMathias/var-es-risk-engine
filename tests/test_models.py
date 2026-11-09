import numpy as np
import pandas as pd
import pytest

from var_es.models import (gaussian_loss_params, simulate_returns, var_es_empirical,
                           var_es_gaussian, var_es_monte_carlo)
from var_es.portfolio import covariance_from_vol_corr, make_weights, portfolio_returns

# Portefeuille stressé du notebook
V0 = 100_000
W = np.array([0.50, 0.30, 0.20])
MU = np.array([0.0005, 0.0003, 0.0004])
VOL = np.array([0.020, 0.015, 0.025])
CORR_STRESS = np.full((3, 3), 0.8) + 0.2 * np.eye(3)
COV_STRESS = covariance_from_vol_corr(VOL, CORR_STRESS)


def test_gaussian_var_matches_notebook_value():
    """La VaR 99 % théorique du portefeuille stressé vaut ~4204,78 € (valeur du notebook)."""
    var, _ = var_es_gaussian(*gaussian_loss_params(MU, COV_STRESS, W, V0), alpha=0.99)
    assert var == pytest.approx(4204.78, abs=0.01)


def test_es_is_larger_than_var():
    for alpha in (0.95, 0.99):
        var, es = var_es_gaussian(*gaussian_loss_params(MU, COV_STRESS, W, V0), alpha)
        assert es > var


def test_standard_normal_closed_form():
    """Pour N(0, 1) : VaR 95 % = 1,645 et ES 95 % = 2,063."""
    var, es = var_es_gaussian(0.0, 1.0, 0.95)
    assert var == pytest.approx(1.6449, abs=1e-3)
    assert es == pytest.approx(2.0627, abs=1e-3)


def test_monte_carlo_close_to_closed_form():
    rng = np.random.default_rng(0)
    var_th, es_th = var_es_gaussian(*gaussian_loss_params(MU, COV_STRESS, W, V0), 0.99)
    var_mc, es_mc = var_es_monte_carlo(MU, COV_STRESS, W, V0, 0.99, n=400_000, rng=rng)
    assert var_mc == pytest.approx(var_th, rel=0.02)
    assert es_mc == pytest.approx(es_th, rel=0.02)


def test_student_simulation_keeps_covariance():
    rng = np.random.default_rng(1)
    x = simulate_returns(np.zeros(3), COV_STRESS, 400_000, rng=rng, dof=6)
    assert np.allclose(np.cov(x.T), COV_STRESS, rtol=0.10)


def test_student_has_fatter_tail_than_gaussian():
    rng = np.random.default_rng(2)
    n = 400_000
    xn = simulate_returns(np.zeros(3), COV_STRESS, n, rng=rng) @ W
    xt = simulate_returns(np.zeros(3), COV_STRESS, n, rng=rng, dof=4) @ W
    assert var_es_empirical(-xt, 0.999)[0] > var_es_empirical(-xn, 0.999)[0]


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        var_es_gaussian(0.0, 1.0, 1.5)
    with pytest.raises(ValueError):
        var_es_empirical([1.0, 2.0], 0.0)


# ---- portefeuille : alignement par ticker --------------------------------
def test_portfolio_alignment_is_by_ticker_not_position():
    """Réordonner les colonnes (comme le fait yfinance) ne doit pas changer le résultat."""
    returns = pd.DataFrame({"SPY": [0.01, -0.02], "QQQ": [0.02, 0.00], "IWM": [-0.01, 0.03]})
    weights = make_weights({"SPY": 0.5, "QQQ": 0.3, "IWM": 0.2})
    expected = 0.5 * returns["SPY"] + 0.3 * returns["QQQ"] + 0.2 * returns["IWM"]
    shuffled = returns[["IWM", "QQQ", "SPY"]]  # ordre alphabétique
    pd.testing.assert_series_equal(portfolio_returns(returns, weights),
                                   expected.rename("portfolio"))
    pd.testing.assert_series_equal(portfolio_returns(shuffled, weights),
                                   expected.rename("portfolio"))


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        make_weights({"SPY": 0.5, "QQQ": 0.3})


def test_missing_ticker_raises():
    returns = pd.DataFrame({"SPY": [0.01]})
    with pytest.raises(KeyError):
        portfolio_returns(returns, make_weights({"SPY": 0.5, "QQQ": 0.5}))

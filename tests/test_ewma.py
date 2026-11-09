import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from var_es.data import synthetic_returns
from var_es.ewma import ewma_var_es, ewma_volatility


@pytest.fixture(scope="module")
def port():
    return synthetic_returns(["SPY"], n_days=900, seed=5)["SPY"]


def test_recursion_matches_manual_computation():
    r = pd.Series([0.01, -0.02, 0.015, 0.0, 0.03, -0.01],
                  index=pd.date_range("2020-01-01", periods=6))
    lam, warmup = 0.9, 3
    s = ewma_volatility(r, lam=lam, warmup=warmup)
    s2_0 = np.var(r.iloc[:3], ddof=1)
    s2_1 = lam * s2_0 + (1 - lam) * r.iloc[3] ** 2
    s2_2 = lam * s2_1 + (1 - lam) * r.iloc[4] ** 2
    assert s.iloc[0] == pytest.approx(np.sqrt(s2_0))
    assert s.iloc[1] == pytest.approx(np.sqrt(s2_1))
    assert s.iloc[2] == pytest.approx(np.sqrt(s2_2))
    assert s.index[0] == r.index[3]


def test_constant_amplitude_converges_to_that_amplitude():
    r = pd.Series(np.tile([0.01, -0.01], 300), index=pd.bdate_range("2020-01-01", periods=600))
    s = ewma_volatility(r, lam=0.94, warmup=30)
    assert s.iloc[-1] == pytest.approx(0.01, rel=1e-6)


def test_volatility_reacts_to_a_shock():
    r = pd.Series(np.full(200, 0.001) * np.tile([1, -1], 100),
                  index=pd.bdate_range("2020-01-01", periods=200))
    r.iloc[100] = 0.08
    s = ewma_volatility(r, lam=0.94, warmup=30)
    assert s.loc[r.index[101]] > 5 * s.loc[r.index[99]]


@pytest.mark.parametrize("method", ["gaussian", "fhs"])
def test_no_look_ahead(port, method):
    """Changer les rendements à partir du jour t ne change pas la prévision pour t."""
    t = 600
    f1 = ewma_var_es(port, 0.99, method=method, start=500)
    altered = port.copy()
    altered.iloc[t:] = altered.iloc[t:] * 10 + 0.5
    f2 = ewma_var_es(altered, 0.99, method=method, start=500)
    day = port.index[t]
    assert f1.loc[day, "var"] == pytest.approx(f2.loc[day, "var"])
    assert f1.index[0] == port.index[500]


@pytest.mark.parametrize("method", ["gaussian", "fhs"])
def test_es_above_var_and_positive(port, method):
    f = ewma_var_es(port, 0.99, method=method)
    assert (f["es"] >= f["var"]).all() and (f["var"] > 0).all()


def test_gaussian_forecast_formula(port):
    f = ewma_var_es(port, 0.95, method="gaussian", lam=0.94, start=500)
    s = ewma_volatility(port, lam=0.94, warmup=30)
    day = f.index[10]
    assert f.loc[day, "var"] == pytest.approx(s.loc[day] * norm.ppf(0.95))


def test_fhs_close_to_gaussian_on_gaussian_data():
    rng = np.random.default_rng(0)
    r = pd.Series(0.01 * rng.standard_normal(4000), index=pd.bdate_range("2000-01-03", periods=4000))
    g = ewma_var_es(r, 0.95, method="gaussian", start=1000)["var"]
    h = ewma_var_es(r, 0.95, method="fhs", start=1000)["var"]
    assert h.mean() == pytest.approx(g.mean(), rel=0.10)


def test_invalid_arguments(port):
    with pytest.raises(ValueError):
        ewma_volatility(port, lam=1.2)
    with pytest.raises(ValueError):
        ewma_var_es(port, 0.95, method="garch")
    with pytest.raises(ValueError):
        ewma_var_es(port, 0.95, start=10, warmup=30)

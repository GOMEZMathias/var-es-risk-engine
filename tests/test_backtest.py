import numpy as np
import pandas as pd
import pytest

from var_es.backtest import (backtest_summary, basel_traffic_light, christoffersen_conditional_coverage,
                             christoffersen_independence, count_transitions, kupiec_pof,
                             rolling_var_es, violations)
from var_es.data import synthetic_returns


def _viol(n, positions):
    v = np.zeros(n, dtype=int)
    v[list(positions)] = 1
    return v


# ---- Kupiec ---------------------------------------------------------------
def test_kupiec_matches_notebook_qqq():
    """7 violations sur 254 jours à 95 % : p-value ~0,0739 (valeur du notebook)."""
    lr, p = kupiec_pof(_viol(254, range(0, 70, 10)), alpha=0.95)
    assert p == pytest.approx(0.0739, abs=5e-4)


def test_kupiec_perfect_coverage_gives_zero_statistic():
    lr, p = kupiec_pof(_viol(200, range(0, 200, 20)), alpha=0.95)  # 10/200 = 5 %
    assert lr == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0)


def test_kupiec_zero_violations_does_not_crash():
    lr, p = kupiec_pof(np.zeros(254, dtype=int), alpha=0.95)
    assert np.isfinite(lr) and 0.0 <= p <= 1.0


def test_kupiec_rejects_too_many_violations():
    _, p = kupiec_pof(_viol(250, range(0, 100, 4)), alpha=0.99)  # 25 violations vs 2,5 attendues
    assert p < 0.001


# ---- Christoffersen -------------------------------------------------------
def test_count_transitions():
    assert count_transitions([0, 0, 1, 1, 0]) == (1, 1, 1, 1)


def test_clustered_violations_fail_independence():
    v = _viol(250, [100, 101, 102, 103, 104, 105, 150, 151, 152, 153, 154, 155])
    _, p = christoffersen_independence(v)
    assert p < 0.05


def test_spread_violations_pass_independence():
    v = _viol(250, range(5, 250, 20))
    _, p = christoffersen_independence(v)
    assert p > 0.05


def test_independence_without_violations_is_zero():
    lr, p = christoffersen_independence(np.zeros(100, dtype=int))
    assert lr == 0.0 and p == 1.0


def test_conditional_coverage_is_sum_of_both_statistics():
    v = _viol(250, [10, 11, 12, 100, 200])
    lr_cc, _ = christoffersen_conditional_coverage(v, 0.95)
    assert lr_cc == pytest.approx(kupiec_pof(v, 0.95)[0] + christoffersen_independence(v)[0])


def test_basel_zones():
    assert [basel_traffic_light(k) for k in (0, 4, 5, 9, 10)] == [
        "vert", "vert", "orange", "orange", "rouge"]


def test_summary_keys():
    s = backtest_summary(_viol(250, [10, 50]), 0.99)
    assert s["violations"] == 2 and s["attendues"] == 2.5 and s["n_obs"] == 250


# ---- prévisions glissantes : pas de look-ahead --------------------------------
@pytest.fixture(scope="module")
def port():
    return synthetic_returns(["SPY"], n_days=700, seed=3)["SPY"]


@pytest.mark.parametrize("method", ["historical", "gaussian"])
@pytest.mark.parametrize("window", [None, 120])
def test_no_look_ahead(port, method, window):
    """Modifier les rendements à partir du jour t ne doit pas changer la VaR prévue pour t."""
    t = 400
    f1 = rolling_var_es(port, 0.95, method=method, window=window, start=250)
    altered = port.copy()
    altered.iloc[t:] = altered.iloc[t:] * 10 + 0.5
    f2 = rolling_var_es(altered, 0.95, method=method, window=window, start=250)
    day = port.index[t]
    assert f1.loc[day, "var"] == pytest.approx(f2.loc[day, "var"])
    assert f1.index[0] == port.index[250 if window is None else max(250, window)]


def test_es_above_var(port):
    f = rolling_var_es(port, 0.99, method="historical", window=250)
    assert (f["es"] >= f["var"]).all()


def test_violations_definition():
    idx = pd.date_range("2020-01-01", periods=4)
    returns = pd.Series([-0.03, 0.01, -0.01, -0.05], index=idx)
    var = pd.Series([0.02, 0.02, 0.02, 0.02], index=idx)
    assert violations(returns, var).tolist() == [True, False, False, True]


def test_window_larger_than_start_raises(port):
    with pytest.raises(ValueError):
        rolling_var_es(port, 0.95, window=300, start=250)

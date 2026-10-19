import numpy as np, pandas as pd
from tscompare.tasks import Task, make_synthetic_task, seasonal_scale, season_length_from_freq, pandas_freq
from tscompare.metrics import evaluate_forecast, wql, QUANTILES
from tscompare.models.baselines import SeasonalNaive, Naive
from tscompare.models.base import clean_context
from tscompare import runner, aggregate as ag


def test_seasonal_scale_hand_calc():
    assert np.isclose(seasonal_scale(np.array([1., 2, 4, 7, 11]), 1), (1 + 2 + 3 + 4) / 4)
    assert np.isnan(seasonal_scale(np.ones(10), 2))


def test_season_length():
    assert season_length_from_freq("h") == 24 and season_length_from_freq("15T") == 96
    assert season_length_from_freq("ME") == 12 and season_length_from_freq("QE") == 4
    assert pandas_freq("5T") == "5min" and pandas_freq("H") == "h"


def test_mase_perfect_and_naive():
    t = make_synthetic_task(n_series=4, seed=1)
    r = evaluate_forecast(t, t.targets)
    assert r["mase"] == 0 and r["mse"] == 0
    r2 = evaluate_forecast(t, t.targets + 1.0)
    assert np.isclose(r2["mae"], 1.0)


def test_wql_pinball():
    y = np.array([[1., 2.]]); q = np.broadcast_to(y[:, None, :], (1, 9, 2)).copy()
    assert wql(y, q) == 0
    q2 = q + 1  # all quantiles over-predict by 1
    exp = np.mean([2 * (1 - tau) * 1 for tau in QUANTILES]) * 2 / 3  # sum over 2 steps / sum|y|=3
    assert np.isclose(wql(y, q2), exp)


def test_clean_context_nan():
    x = np.array([np.nan, 1, np.nan, 3, np.nan])
    assert np.allclose(clean_context(x), [1, 1, 1, 3, 3])


def test_no_leak_synthetic_split():
    t = make_synthetic_task(n_series=3, length=300, horizon=12, seed=2)
    # train series must end strictly before first test target
    assert all(len(s) < len(c) for s, c in zip(t.train_series, t.contexts[::2]))


def test_runner_and_aggregate(tmp_path):
    tasks = [make_synthetic_task(name=f"syn/{i}", seed=i, n_series=6) for i in range(8)]
    for m in (SeasonalNaive(), Naive()):
        runner.run(m, tasks, out_dir=tmp_path, verbose=False)
    df = ag.load_results(tmp_path)
    assert (df.status == "ok").all() and len(df) == 16
    tbl = ag.summary_table(df)
    assert np.isclose(tbl.loc["seasonal_naive", "mase_rel"], 1.0)
    assert tbl.loc["seasonal_naive", "win_rate_%"] + tbl.loc["naive", "win_rate_%"] == 100
    r = ag.friedman_nemenyi(ag.wide(df))
    assert r["cd"] > 0


def test_failure_imputation(tmp_path):
    from tscompare.models.base import Forecaster, Forecast
    class Broken(Forecaster):
        name = "broken"
        def predict(self, task): raise RuntimeError("boom")
    tasks = [make_synthetic_task(name=f"syn/{i}", seed=i, n_series=4) for i in range(3)]
    runner.run(SeasonalNaive(), tasks, tmp_path, verbose=False); runner.run(Broken(), tasks, tmp_path, verbose=False)
    df = ag.load_results(tmp_path); tbl = ag.summary_table(df)
    assert tbl.loc["broken", "tasks_failed"] == 3 and np.isclose(tbl.loc["broken", "mase_rel"], 1.0)

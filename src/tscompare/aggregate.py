"""Leaderboard-style aggregation (GIFT-Eval / fev-bench conventions).

relative error  = metric(model, task) / metric(seasonal_naive, task)
headline score  = geometric mean of relative error across tasks (lower better)
skill score     = 1 - geometric mean relative error (higher better; fev-bench)
rank            = per-task rank (1 best), averaged over tasks
win rate        = fraction of (task, opponent) pairs won, ties = 0.5
Failed / unsupported tasks are imputed with seasonal naive (fev-bench rule) unless impute=False.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = "seasonal_naive"


def load_results(root="results") -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in Path(root).rglob("*.json")]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop(columns=[c for c in ("trace", "model_info") if c in df.columns])


def wide(df: pd.DataFrame, metric="mase", impute=True, models=None) -> pd.DataFrame:
    """tasks x models matrix of `metric`.
    Failed/unsupported (task, model) cells are filled with seasonal naive when impute=True, else NaN.
    A model that structurally lacks the metric (e.g. no quantiles -> no WQL) stays NaN: it is NOT imputed."""
    w = df.pivot_table(index="task", columns="model", values=metric, aggfunc="first", dropna=False)
    st = df.pivot_table(index="task", columns="model", values="status", aggfunc="first", dropna=False)
    w = w.where(st == "ok")
    if BASE in w.columns and impute:
        for m in w.columns:
            bad = st[m].notna() & (st[m] != "ok")
            w.loc[bad, m] = w.loc[bad, BASE]
    if models is not None:
        w = w[[m for m in models if m in w.columns]]
    return w


def geo_mean(x):
    x = np.asarray(x, float); x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if x.size else np.nan


def bootstrap_gm(rel: pd.Series, n=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed); v = rel.dropna().values
    if v.size < 2:
        return (np.nan, np.nan)
    bs = [geo_mean(rng.choice(v, v.size)) for _ in range(n)]
    return tuple(np.quantile(bs, [alpha / 2, 1 - alpha / 2]))


def pairwise_winrate(w: pd.DataFrame) -> pd.DataFrame:
    ms = list(w.columns); out = pd.DataFrame(index=ms, columns=ms, dtype=float)
    for a in ms:
        for b in ms:
            if a == b: out.loc[a, b] = np.nan; continue
            ok = w[a].notna() & w[b].notna()
            out.loc[a, b] = ((w.loc[ok, a] < w.loc[ok, b]).sum() + 0.5 * (w.loc[ok, a] == w.loc[ok, b]).sum()) / max(ok.sum(), 1)
    return out


def summary_table(df: pd.DataFrame, metrics=("mase", "wql"), impute=True, models=None) -> pd.DataFrame:
    """One row per model: relative GM per metric (+95% CI for first), skill, avg rank, win rate, coverage, time."""
    res = {}
    for k, met in enumerate(metrics):
        w = wide(df, met, impute, models)
        rel = w.div(w[BASE], axis=0) if BASE in w else w
        res[f"{met}_rel"] = rel.apply(geo_mean)
        if k == 0:
            ci = {m: bootstrap_gm(rel[m]) for m in rel.columns}
            res[f"{met}_ci_lo"] = pd.Series({m: v[0] for m, v in ci.items()})
            res[f"{met}_ci_hi"] = pd.Series({m: v[1] for m, v in ci.items()})
            res["skill_%"] = 100 * (1 - res[f"{met}_rel"])
            res["avg_rank"] = w.rank(axis=1).mean()
            res["win_rate_%"] = 100 * pairwise_winrate(w).mean(axis=1)
    ok = df[df.status == "ok"]
    res["tasks_ok"] = ok.groupby("model").task.nunique()
    res["tasks_failed"] = df[df.status == "failed"].groupby("model").task.nunique()
    res["tasks_unsupported"] = df[df.status == "unsupported"].groupby("model").task.nunique()
    res["fit_s_median"] = ok.groupby("model").fit_s.median()
    res["predict_s_median"] = ok.groupby("model").predict_s.median()
    out = pd.DataFrame(res).fillna({"tasks_failed": 0, "tasks_unsupported": 0})
    fam = df.drop_duplicates("model").set_index("model")[["family", "track"]]
    return fam.join(out).sort_values(f"{metrics[0]}_rel")


def group_table(df: pd.DataFrame, by: str, metric="mase", impute=True) -> pd.DataFrame:
    """Geometric-mean relative error per model within each group of `by` (domain, freq, term, ...)."""
    meta = df.drop_duplicates("task").set_index("task")[by]
    w = wide(df, metric, impute); rel = w.div(w[BASE], axis=0)
    return rel.groupby(meta.reindex(rel.index)).agg(geo_mean).T


def friedman_nemenyi(w: pd.DataFrame, alpha=0.05):
    """Friedman test + Nemenyi critical difference on a tasks x models matrix (complete cases)."""
    w = w.dropna(); N, k = w.shape
    chi = stats.friedmanchisquare(*[w[c] for c in w.columns]) if k >= 3 and N >= 2 else None
    ranks = w.rank(axis=1).mean()
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    return dict(avg_ranks=ranks.sort_values(), cd=float(cd), n_tasks=N, k=k,
                friedman_p=None if chi is None else float(chi.pvalue))


def paired_wilcoxon(w: pd.DataFrame, a: str, b: str):
    ok = w[[a, b]].dropna()
    if len(ok) < 6: return np.nan
    return float(stats.wilcoxon(np.log(ok[a]), np.log(ok[b])).pvalue)


def to_latex(tbl: pd.DataFrame, cols, fmt="{:.3f}", caption="", label="") -> str:
    t = tbl[cols].copy()
    for c in cols:
        best = t[c].min() if "rel" in c or "rank" in c else t[c].max()
        t[c] = t[c].map(lambda v, c=c, best=best: (r"\textbf{" + fmt.format(v) + "}") if np.isclose(v, best) else fmt.format(v))
    return t.to_latex(escape=True, column_format="l" + "r" * len(cols), caption=caption or None, label=label or None).replace(r"\textbackslash{}textbf\{", r"\textbf{")

"""Figures for the paper. Each returns a matplotlib Figure; save with fig.savefig(..., bbox_inches='tight')."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..aggregate import BASE, friedman_nemenyi, geo_mean, wide

FAMILY_COLORS = {"baseline": "#8c8c8c", "pre-llm": "#4c72b0", "llm-reuse": "#dd8452",
                 "native-tsfm": "#55a868", "critical": "#c44e52"}


def bar_relative(tbl: pd.DataFrame, col="mase_rel", lo="mase_ci_lo", hi="mase_ci_hi", title=""):
    t = tbl.sort_values(col); fig, ax = plt.subplots(figsize=(7, 0.38 * len(t) + 1))
    y = np.arange(len(t))
    ax.barh(y, t[col], color=[FAMILY_COLORS.get(f, "#999") for f in t.family],
            xerr=[t[col] - t[lo], t[hi] - t[col]] if lo in t else None, capsize=2)
    ax.axvline(1, color="k", ls="--", lw=.8); ax.set_yticks(y, t.index); ax.invert_yaxis()
    ax.set_xlabel("Geometric-mean relative MASE (vs seasonal naive, lower is better)"); ax.set_title(title)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f) for f, c in FAMILY_COLORS.items() if f in set(t.family)],
              loc="lower right", fontsize=8)
    return fig


def critical_difference(w: pd.DataFrame, alpha=0.05, title="Critical difference (Nemenyi)"):
    r = friedman_nemenyi(w, alpha); ranks = r["avg_ranks"]; cd = r["cd"]; k = len(ranks)
    half = k // 2 + k % 2
    fig, ax = plt.subplots(figsize=(8, 0.3 * half + 2.2)); lo, hi = 1, k
    ax.set_xlim(lo - 1.6, hi + 1.6); ax.set_ylim(-0.5 * half - 0.6, 2.3); ax.axis("off")
    ax.hlines(1, lo, hi, color="k"); [ax.text(i, 1.1, str(i), ha="center", fontsize=8) for i in range(lo, hi + 1)]
    for j, (m, rk) in enumerate(ranks.items()):
        left = j < half; yy = 0.45 - 0.5 * (j if left else j - half)
        x_end = lo - .15 if left else hi + .15
        ax.plot([rk, rk, x_end], [1, yy, yy], color="#444", lw=.7)
        ax.text(x_end + (-.05 if left else .05), yy, f"{m} ({rk:.2f})", ha="right" if left else "left", va="center", fontsize=8)
    ax.hlines(1.8, lo, lo + cd, color="r", lw=3); ax.text(lo + cd / 2, 1.9, f"CD = {cd:.2f}", ha="center", color="r", fontsize=8)
    items = list(ranks.items()); cl = []
    for i, (_, rk) in enumerate(items):
        j = max(t for t, (_, r2) in enumerate(items) if r2 - rk <= cd)
        if j > i: cl.append((i, j))
    cl = [c for c in cl if not any(o != c and o[0] <= c[0] and o[1] >= c[1] for o in cl)]
    for n, (i, j) in enumerate(cl):
        ax.hlines(0.8 - 0.09 * n, items[i][1] - .04, items[j][1] + .04, color="k", lw=2.5)
    t = f"{title}  [N={r['n_tasks']} tasks" + (f", Friedman p={r['friedman_p']:.2g}]" if r["friedman_p"] is not None else "]")
    ax.set_title(t, fontsize=9, pad=6)
    return fig


def heatmap_tasks(df, metric="mase", impute=True, title=""):
    w = wide(df, metric, impute); rel = np.log2(w.div(w[BASE], axis=0))
    fig, ax = plt.subplots(figsize=(0.6 * rel.shape[1] + 3, 0.28 * rel.shape[0] + 2))
    im = ax.imshow(rel.values, cmap="RdYlGn_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(rel.shape[1]), rel.columns, rotation=60, ha="right"); ax.set_yticks(range(rel.shape[0]), rel.index, fontsize=6)
    fig.colorbar(im, label="log2(rel. MASE vs seasonal naive)"); ax.set_title(title); return fig


def group_heatmap(gt: pd.DataFrame, title=""):
    fig, ax = plt.subplots(figsize=(0.9 * gt.shape[1] + 3, 0.35 * gt.shape[0] + 1.5))
    im = ax.imshow(np.log2(gt.values.astype(float)), cmap="RdYlGn_r", vmin=-1.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(gt.shape[1]), gt.columns, rotation=30, ha="right"); ax.set_yticks(range(gt.shape[0]), gt.index)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            ax.text(j, i, f"{gt.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="log2(rel. MASE)"); ax.set_title(title); return fig


def accuracy_vs_cost(tbl: pd.DataFrame, cost_col="predict_s_median", acc_col="mase_rel", title=""):
    t = tbl.dropna(subset=[cost_col, acc_col]); fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m, r in t.iterrows():
        ax.scatter(r[cost_col], r[acc_col], c=FAMILY_COLORS.get(r.family, "#999"), s=60)
        ax.annotate(m, (r[cost_col], r[acc_col]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("median inference time per task (s, log)"); ax.set_ylabel("rel. MASE (lower better)")
    ax.axhline(1, color="k", ls="--", lw=.7); ax.set_title(title); return fig


def horizon_curve(npz_by_model: dict, targets: np.ndarray, scale: np.ndarray):
    """Error vs forecast step. npz_by_model: {model: point (n,H)}."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m, p in npz_by_model.items():
        e = np.nanmean(np.abs(targets - p) / scale[:, None], axis=0); ax.plot(np.arange(1, len(e) + 1), e, label=m)
    ax.set_xlabel("forecast step"); ax.set_ylabel("scaled abs. error"); ax.legend(fontsize=7); return fig

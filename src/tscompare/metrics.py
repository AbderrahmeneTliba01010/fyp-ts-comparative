"""Metrics. All functions are NaN-aware and vectorised over instances.

point forecast used for MASE/MAE/MSE = median (q=0.5) when quantiles exist, else the model's point output.
WQL (a.k.a. mean weighted quantile loss, the CRPS proxy used by GIFT-Eval / fev-bench) is pooled per task.
"""
from __future__ import annotations

import numpy as np

from .tasks import Task, seasonal_scale

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def mase_per_instance(y, yhat, scale):
    err = np.nanmean(np.abs(y - yhat), axis=1)
    out = err / scale
    out[~np.isfinite(scale)] = np.nan
    return out


def wql(y, q, levels=QUANTILES):
    """y: (n,H); q: (n,Q,H). Returns pooled weighted quantile loss (scalar)."""
    levels = np.asarray(levels)[None, :, None]
    yy = y[:, None, :]
    diff = yy - q
    loss = 2 * np.abs(diff * ((yy <= q).astype(float) - levels))
    num = np.nansum(loss) / q.shape[1]
    den = np.nansum(np.abs(y))
    return float(num / den) if den > 0 else np.nan


def _ctx_stats(contexts, window=None):
    mu = np.array([np.nanmean(c[-window:] if window else c) for c in contexts])
    sd = np.array([np.nanstd(c[-window:] if window else c) for c in contexts])
    sd = np.where((~np.isfinite(sd)) | (sd == 0), 1.0, sd)
    return mu, sd


def evaluate_forecast(task: Task, point: np.ndarray, quantiles: np.ndarray | None = None,
                      levels=QUANTILES, norm_window: int = 512) -> dict:
    y = task.targets.astype(float)
    p = np.asarray(point, dtype=float)
    assert p.shape == y.shape, f"point shape {p.shape} != target {y.shape}"
    scale = task.mase_scale
    if scale is None:
        scale = np.array([seasonal_scale(c, task.season_length) for c in task.contexts])
    inst_mase = mase_per_instance(y, p, scale)
    mu, sd = _ctx_stats(task.contexts, norm_window)
    z = (y - mu[:, None]) / sd[:, None]
    zh = (p - mu[:, None]) / sd[:, None]
    out = dict(
        mase=float(np.nanmean(inst_mase)),
        mae=float(np.nanmean(np.abs(y - p))),
        mse=float(np.nanmean((y - p) ** 2)),
        nmae=float(np.nanmean(np.abs(z - zh))),
        nmse=float(np.nanmean((z - zh) ** 2)),
        smape=float(np.nanmean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-8))),
        n_instances=int(y.shape[0]),
        n_mase_valid=int(np.isfinite(inst_mase).sum()),
    )
    out["wql"] = wql(y, np.asarray(quantiles, dtype=float), levels) if quantiles is not None else np.nan
    return out

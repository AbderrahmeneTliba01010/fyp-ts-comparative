"""Mechanism probe tied to the FYP hypothesis (slide 15): do coarse/quantised/frozen-LLM methods lose
accuracy specifically around sharp local dynamics (spikes, turning points)?  Needs saved predictions."""
from __future__ import annotations

import numpy as np


def local_dynamics_masks(targets: np.ndarray, scale: np.ndarray, top_frac=0.10):
    """Boolean (n,H) mask of the top `top_frac` steps by |second difference| / scale (pooled per task)."""
    d2 = np.abs(np.diff(targets, n=2, axis=1)); d2 = np.pad(d2, ((0, 0), (1, 1)), mode="edge")
    z = d2 / scale[:, None]
    thr = np.nanquantile(z[np.isfinite(z)], 1 - top_frac)
    return z >= thr


def error_by_regime(targets, point, scale, top_frac=0.10) -> dict:
    """Mean scaled abs error on high-curvature steps vs the rest; ratio>1 means worse on sharp dynamics."""
    mask = local_dynamics_masks(targets, scale, top_frac)
    e = np.abs(targets - point) / scale[:, None]
    hi, lo = np.nanmean(e[mask]), np.nanmean(e[~mask])
    return dict(err_high_curv=float(hi), err_low_curv=float(lo), ratio=float(hi / lo))

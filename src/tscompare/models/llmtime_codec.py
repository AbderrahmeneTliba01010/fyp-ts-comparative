"""LLMTime (Gruver et al., 2023) numeric <-> text codec. Pure numpy so it is unit-testable without a GPU.

scale -> fixed-precision integers -> digits separated by spaces -> steps separated by ','."""
from __future__ import annotations

import numpy as np


def fit_scaler(history, alpha=0.95, beta=0.3):
    h = history[np.isfinite(history)]
    lo, hi = h.min(), h.max()
    min_ = lo - beta * (hi - lo)
    q = np.quantile(h - min_, alpha)
    q = q if q > 0 else 1.0
    return min_, q


def encode(values, min_, q, prec=3, width=None):
    """values -> string like '0 1 2 , 0 1 5 , ...' (fixed width digits, no decimal point)."""
    v = np.rint((np.asarray(values) - min_) / q * 10 ** prec).astype(np.int64)
    v = np.clip(v, 0, None)
    width = width or max(len(str(int(v.max()))), prec + 1)
    v = np.clip(v, 0, 10 ** width - 1)
    return " , ".join(" ".join(str(x).zfill(width)) for x in v), width


def decode(text, min_, q, prec=3, width=4, horizon=None):
    """Parse generated text back to floats; invalid steps -> NaN. Returns (horizon,) array."""
    steps = [s.replace(" ", "") for s in text.split(",")]
    out = []
    for s in steps:
        out.append(int(s) / 10 ** prec * q + min_ if s.isdigit() and len(s) == width else np.nan)
    out = np.array(out, dtype=float)
    if horizon is not None:
        out = np.pad(out, (0, max(0, horizon - len(out)), ), constant_values=np.nan)[:horizon]
    return out

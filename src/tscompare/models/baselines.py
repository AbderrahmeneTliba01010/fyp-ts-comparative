"""Statistical references. Seasonal naive is the normaliser for every relative metric."""
from __future__ import annotations

import numpy as np

from ..metrics import QUANTILES
from ..tasks import Task
from .base import Forecast, Forecaster, clean_context


def _seasonal_point(c: np.ndarray, H: int, m: int) -> np.ndarray:
    m = max(1, min(m, len(c)))
    reps = int(np.ceil(H / m))
    return np.tile(c[-m:], reps)[:H]


def _empirical_quantiles(point, resid, H):
    """Quantiles = point + empirical in-context residual quantiles, widened by sqrt(1 + h/H)."""
    if resid.size < 8:
        resid = np.zeros(8)
    rq = np.quantile(resid, QUANTILES)                 # (Q,)
    widen = np.sqrt(1 + np.arange(H) / max(H, 1))      # (H,)
    return point[None, :] + rq[:, None] * widen[None, :]


class SeasonalNaive(Forecaster):
    name, family, track = "seasonal_naive", "baseline", "zeroshot"

    def predict(self, task: Task) -> Forecast:
        H, m = task.horizon, task.season_length
        P, Q = [], []
        for c in task.contexts:
            c = clean_context(c, self.context_length * 4)
            p = _seasonal_point(c, H, m)
            mm = max(1, min(m, len(c) - 1))
            resid = (c[mm:] - c[:-mm])[-512:]
            P.append(p); Q.append(_empirical_quantiles(p, resid, H))
        return Forecast(np.stack(P), np.stack(Q))


class Naive(Forecaster):
    name, family, track = "naive", "baseline", "zeroshot"

    def predict(self, task: Task) -> Forecast:
        H = task.horizon
        P, Q = [], []
        for c in task.contexts:
            c = clean_context(c, self.context_length * 4)
            p = np.full(H, c[-1], dtype=np.float32)
            Q.append(_empirical_quantiles(p, np.diff(c)[-512:], H)); P.append(p)
        return Forecast(np.stack(P), np.stack(Q))


class SeasonalAvg(Forecaster):
    """Average of the last k seasons (k=4): a stronger statistical reference than seasonal naive on noisy data."""
    name, family, track = "seasonal_avg", "baseline", "zeroshot"

    def predict(self, task: Task) -> Forecast:
        H, m, k = task.horizon, task.season_length, 4
        P, Q = [], []
        for c in task.contexts:
            c = clean_context(c, self.context_length * 4); mm = max(1, min(m, len(c)))
            kk = max(1, min(k, len(c) // mm)); seas = c[len(c) - kk * mm:].reshape(kk, mm).mean(0)
            p = np.tile(seas, int(np.ceil(H / mm)))[:H]
            fit = np.tile(seas, kk); resid = (c[len(c) - kk * mm:] - fit)[-512:]
            P.append(p); Q.append(_empirical_quantiles(p, resid, H))
        return Forecast(np.stack(P), np.stack(Q))


class Drift(Forecaster):
    name, family, track = "drift", "baseline", "zeroshot"

    def predict(self, task: Task) -> Forecast:
        H = task.horizon; P, Q = [], []
        for c in task.contexts:
            c = clean_context(c, self.context_length * 4); slope = (c[-1] - c[0]) / max(len(c) - 1, 1)
            p = c[-1] + slope * np.arange(1, H + 1, dtype=np.float32)
            P.append(p); Q.append(_empirical_quantiles(p, np.diff(c)[-512:] - slope, H))
        return Forecast(np.stack(P), np.stack(Q))

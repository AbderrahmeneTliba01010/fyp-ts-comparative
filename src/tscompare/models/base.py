"""Forecaster interface. One class per method; wrappers import heavy deps lazily."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..metrics import QUANTILES
from ..tasks import Task


@dataclass
class Forecast:
    point: np.ndarray                       # (n, H)
    quantiles: Optional[np.ndarray] = None  # (n, Q, H) at QUANTILES


def clean_context(x: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
    """Truncate to max_len, forward/back-fill NaN, zeros if all-NaN. Returns float32."""
    x = np.asarray(x, dtype=np.float32)
    if max_len:
        x = x[-max_len:]
    if np.isnan(x).any():
        idx = np.where(~np.isnan(x), np.arange(len(x)), 0)
        np.maximum.accumulate(idx, out=idx)
        x = x[idx]
        if np.isnan(x).any():
            fv = x[~np.isnan(x)]
            x = np.nan_to_num(x, nan=float(fv[0]) if fv.size else 0.0)
    return x


def samples_to_forecast(samples: np.ndarray) -> Forecast:
    """samples: (n, S, H) -> median point + QUANTILES."""
    q = np.quantile(samples, QUANTILES, axis=1).transpose(1, 0, 2)
    return Forecast(point=q[:, QUANTILES.index(0.5)], quantiles=q)


class Forecaster(ABC):
    name: str = "base"
    family: str = "baseline"   # baseline | pre-llm | llm-reuse | native-tsfm | critical
    track: str = "zeroshot"    # zeroshot | supervised
    needs_gpu: bool = False

    def __init__(self, context_length: int = 512, **kw):
        self.context_length = context_length
        self.kw = kw

    def supports(self, task: Task) -> bool:
        return True

    def fit(self, task: Task) -> None:  # supervised track only
        pass

    @abstractmethod
    def predict(self, task: Task) -> Forecast:
        """Forecast task.horizon steps for every context in task.contexts."""

    def info(self) -> dict:
        return dict(name=self.name, family=self.family, track=self.track,
                    context_length=self.context_length, **{k: v for k, v in self.kw.items()
                                                           if isinstance(v, (str, int, float, bool))})

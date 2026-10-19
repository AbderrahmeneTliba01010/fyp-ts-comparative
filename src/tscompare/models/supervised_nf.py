"""Informer / Autoformer / FEDformer / PatchTST through Nixtla's `neuralforecast` (one unified, maintained API).

Mode: channel-independent univariate instances, integer time index, no calendar covariates (the original
Informer/Autoformer/FEDformer papers feed timestamp embeddings; this is a documented deviation). Loss: MAE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..tasks import Task
from .base import Forecast, Forecaster, clean_context

ARCH_FAMILY = {"informer": "pre-llm", "autoformer": "pre-llm", "fedformer": "pre-llm", "patchtst": "pre-llm"}


class NFModel(Forecaster):
    track, needs_gpu = "supervised", True

    def __init__(self, arch: str, max_steps=1500, batch_size=32, seed=0, **kw):
        super().__init__(**{k: v for k, v in kw.items() if k == "context_length"})
        self.arch, self.name, self.family = arch, arch, ARCH_FAMILY[arch]
        self.max_steps, self.bs, self.seed = max_steps, batch_size, seed
        self.extra = {k: v for k, v in kw.items() if k != "context_length"}; self.nf = None

    def _long(self, arrs, L=None):
        rows = []
        for i, a in enumerate(arrs):
            a = clean_context(a, L)
            rows.append(pd.DataFrame({"unique_id": f"{i:07d}", "ds": np.arange(len(a)), "y": a.astype(np.float32)}))
        return pd.concat(rows, ignore_index=True)

    def fit(self, task: Task):
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MAE
        from neuralforecast.models import Autoformer, FEDformer, Informer, PatchTST
        cls = dict(informer=Informer, autoformer=Autoformer, fedformer=FEDformer, patchtst=PatchTST)[self.arch]
        H = task.horizon
        ser = [s for s in task.train_series if len(s) >= 3 * H + 16] or task.train_series
        med = int(np.median([len(s) for s in ser]))
        self.L = int(max(16, min(self.context_length, med - 2 * H)))
        kw = dict(h=H, input_size=self.L, loss=MAE(), max_steps=self.max_steps, batch_size=self.bs,
                  val_check_steps=100, early_stop_patience_steps=5, scaler_type="standard", random_seed=self.seed)
        kw.update(self.extra)
        self.nf = NeuralForecast(models=[cls(**kw)], freq=1)
        self.nf.fit(df=self._long(ser, 20000), val_size=H)

    def predict(self, task: Task) -> Forecast:
        df = self._long(task.contexts, self.L)
        out = self.nf.predict(df=df).sort_values(["unique_id", "ds"])
        col = [c for c in out.columns if c not in ("unique_id", "ds")][0]
        return Forecast(out[col].to_numpy(np.float32).reshape(task.n, task.horizon), None)

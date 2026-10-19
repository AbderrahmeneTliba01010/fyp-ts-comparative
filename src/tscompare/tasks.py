"""Task abstraction + data sources (synthetic for smoke tests, TIME benchmark for real runs).

Design decisions (documented in docs/PROTOCOL.md):
  * Everything is channel-independent / univariate: one *instance* = one series (variate) window.
  * An instance is (context, target). Contexts are stored left-truncated to `max_context_store`
    points, but the MASE scale is computed on the FULL context before truncation.
  * Supervised methods may only use `train_series` (and `val_*` for early stopping).
    Test targets are never visible to `fit`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- seasonality
def season_length_from_freq(freq: str) -> int:
    """Fallback seasonal period when gluonts is unavailable (D -> 7, differs from gluonts' 1)."""
    m = re.fullmatch(r"\s*(\d*)\s*([A-Za-z]+)(?:-[A-Za-z]+)?\s*", str(freq))
    if not m:
        return 1
    mult = int(m.group(1) or 1)
    unit = m.group(2).lower()
    if unit in {"s", "sec"}:
        base = 60
    elif unit in {"t", "min"}:
        base = 24 * 60
    elif unit in {"h"}:
        base = 24
    elif unit in {"d", "b"}:
        base = 7 if unit == "d" else 5
    elif unit in {"w"}:
        base = 1
    elif unit in {"m", "me", "ms"}:
        base = 12
    elif unit in {"q", "qe", "qs"}:
        base = 4
    else:  # a, y, ye, ...
        base = 1
    s, rem = divmod(base, mult)
    return s if (rem == 0 and s >= 1) else 1


def pandas_freq(freq: str) -> str:
    """Normalise legacy aliases (T->min, H->h, M->ME, ...) for pandas >= 2.2."""
    m = re.fullmatch(r"\s*(\d*)\s*([A-Za-z]+)(-[A-Za-z]+)?\s*", str(freq))
    if not m:
        return str(freq)
    n, unit, suffix = m.group(1), m.group(2), m.group(3) or ""
    mapping = {"T": "min", "H": "h", "S": "s", "M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}
    return f"{n}{mapping.get(unit, unit)}{suffix}"


# --------------------------------------------------------------------------- Task
@dataclass
class Task:
    name: str                      # e.g. "bitbrains_rnd/5T"
    freq: str
    term: str                      # short | medium | long | synthetic
    horizon: int
    season_length: int
    contexts: list                 # list[np.ndarray] 1-D, may contain NaN (left-truncated)
    targets: np.ndarray            # (n, H)
    mase_scale: Optional[np.ndarray] = None   # (n,) from full context; None -> computed on stored context
    item_ids: Optional[list] = None
    train_series: Optional[list] = None       # list[np.ndarray]; supervised track only
    val_contexts: Optional[list] = None
    val_targets: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)  # domain, n_series, windows, contaminated, ...

    @property
    def slug(self) -> str:
        return f"{self.name}/{self.term}".replace("/", "__")

    @property
    def n(self) -> int:
        return len(self.contexts)

    def subsample(self, n: int, seed: int = 0) -> "Task":
        """Deterministic subsample of evaluation instances (same indices for every model)."""
        if n is None or n >= self.n:
            return self
        idx = np.sort(np.random.default_rng(seed).choice(self.n, size=n, replace=False))
        return Task(
            name=self.name, freq=self.freq, term=self.term, horizon=self.horizon,
            season_length=self.season_length,
            contexts=[self.contexts[i] for i in idx],
            targets=self.targets[idx],
            mase_scale=None if self.mase_scale is None else self.mase_scale[idx],
            item_ids=None if self.item_ids is None else [self.item_ids[i] for i in idx],
            train_series=self.train_series, val_contexts=self.val_contexts,
            val_targets=self.val_targets,
            meta={**self.meta, "subsampled_from": self.n, "subsample_seed": seed},
        )

    def summary(self) -> dict:
        lens = [len(c) for c in self.contexts]
        return dict(task=self.slug, freq=self.freq, term=self.term, horizon=self.horizon,
                    season=self.season_length, n_instances=self.n,
                    ctx_median=int(np.median(lens)), ctx_min=int(min(lens)),
                    n_train=0 if self.train_series is None else len(self.train_series),
                    **{k: v for k, v in self.meta.items() if isinstance(v, (str, int, float, bool))})


def seasonal_scale(context: np.ndarray, m: int) -> float:
    """Mean absolute seasonal difference (the MASE denominator). NaN if undefined/zero."""
    c = np.asarray(context, dtype=float)
    m = max(1, min(int(m), max(len(c) - 1, 1)))
    d = np.abs(c[m:] - c[:-m])
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan
    s = d.mean()
    return float(s) if s > 0 else np.nan


# --------------------------------------------------------------------------- synthetic source
def make_synthetic_task(name="synthetic/H", n_series=24, length=900, horizon=24, season=24,
                        n_windows=2, seed=0, spike_prob=0.01, nan_frac=0.0, freq="h") -> Task:
    """Series = trend + seasonality + AR(1) noise + occasional spikes. For smoke tests only."""
    rng = np.random.default_rng(seed)
    full = []
    t = np.arange(length + n_windows * horizon)
    for _ in range(n_series):
        amp = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        x = (rng.normal(0, 0.002) * t + amp * np.sin(2 * np.pi * t / season + phase)
             + 0.3 * amp * np.sin(2 * np.pi * t / (season * 7) + phase))
        e = np.zeros_like(t, dtype=float)
        for i in range(1, len(t)):
            e[i] = 0.6 * e[i - 1] + rng.normal(0, 0.15)
        x = x + e + 10 * rng.uniform(0.5, 2)
        spikes = rng.random(len(t)) < spike_prob
        x[spikes] += rng.normal(0, 3 * amp, spikes.sum())
        if nan_frac:
            x[rng.random(len(t)) < nan_frac] = np.nan
        full.append(x.astype(np.float32))
    ctxs, tgts, scales, ids = [], [], [], []
    for sid, x in enumerate(full):
        for w in range(n_windows):
            end = length + w * horizon
            ctxs.append(x[:end].copy())
            tgts.append(x[end:end + horizon].copy())
            scales.append(seasonal_scale(x[:end], season))
            ids.append(f"s{sid}_w{w}")
    train = [x[: length - 2 * horizon].copy() for x in full]
    vctx = [x[: length - horizon].copy() for x in full]
    vtgt = np.stack([x[length - horizon: length] for x in full])
    return Task(name=name, freq=freq, term="synthetic", horizon=horizon, season_length=season,
                contexts=ctxs, targets=np.stack(tgts), mase_scale=np.array(scales), item_ids=ids,
                train_series=train, val_contexts=vctx, val_targets=vtgt,
                meta=dict(domain="synthetic", contaminated=False))


# --------------------------------------------------------------------------- TIME benchmark
class TimeBenchSource:
    """Adapter over the official `timebench` package (https://github.com/zqiao11/TIME).

    Uses the package's own `Dataset` / config so that windows, horizons and splits are identical to
    the published leaderboard. Needs: `pip install -e <TIME repo>` and the HF dataset downloaded
    (set `storage_path`, or TIME_DATASET in .env).

    Split semantics (from timebench.evaluation.data.Dataset):
      * test_data           -> `windows` rolling windows over the last `test_length` points
      * training_dataset    -> series up to  T - (windows+1)*H
      * validation_dataset  -> series up to  T -  windows*H   (its last H points = validation target)
    We fit on training_dataset and early-stop on the validation window; test targets are never seen.
    """

    TERMS = ("short", "medium", "long")

    def __init__(self, storage_path: str, config_path: Optional[str] = None):
        try:
            from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install the TIME codebase first: git clone https://github.com/zqiao11/TIME "
                              "&& pip install -e TIME") from e
        self._Dataset, self._settings = Dataset, get_dataset_settings
        from pathlib import Path
        self.storage_path = str(storage_path)
        self.cfg = load_dataset_config(Path(config_path) if config_path else None)

    def list_tasks(self) -> pd.DataFrame:
        rows = []
        for name, dcfg in self.cfg.get("datasets", {}).items():
            for term in self.TERMS:
                if term in dcfg:
                    s = self._settings(name, term, self.cfg)
                    rows.append(dict(name=name, dataset=name.split("/")[0], freq=name.split("/")[-1],
                                     term=term, domain=dcfg.get("domain"), **s))
        return pd.DataFrame(rows)

    def load(self, name: str, term: str, max_instances: Optional[int] = None,
             max_train_series: Optional[int] = 500, max_context_store: int = 4096,
             max_train_len: int = 100_000, seed: int = 0, need_train: bool = True) -> Task:
        D = self._Dataset
        probe = D(name=name, term=term, to_univariate=False, storage_path=self.storage_path)
        to_uni = probe.target_dim != 1          # same guard as the official examples
        s = self._settings(name, term, self.cfg)
        ds = D(name=name, term=term, to_univariate=to_uni, prediction_length=s["prediction_length"],
               test_length=s["test_length"], val_length=s["val_length"] or 0,
               storage_path=self.storage_path)
        H = ds.prediction_length
        try:
            from gluonts.time_feature import get_seasonality
            m = int(get_seasonality(ds.freq))
        except Exception:
            m = season_length_from_freq(ds.freq)

        td = ds.test_data
        ctxs, tgts, scales, ids = [], [], [], []
        for inp, lab in zip(td.input, td.label):
            full = np.asarray(inp["target"], dtype=np.float32)
            scales.append(seasonal_scale(full, m))
            ctxs.append(full[-max_context_store:])
            tgts.append(np.asarray(lab["target"], dtype=np.float32)[:H])
            ids.append(inp.get("item_id"))
        task = Task(name=name, freq=ds.freq, term=term, horizon=H, season_length=m,
                    contexts=ctxs, targets=np.stack(tgts), mase_scale=np.array(scales), item_ids=ids,
                    meta=dict(domain=(self.cfg["datasets"][name].get("domain")), windows=ds.windows,
                              n_series=len(set(map(str, ids))), contaminated=False))
        if max_instances:
            task = task.subsample(max_instances, seed)
        if need_train:
            rng = np.random.default_rng(seed)
            tr = [np.asarray(e["target"], dtype=np.float32)[-max_train_len:] for e in ds.training_dataset]
            if max_train_series and len(tr) > max_train_series:
                tr = [tr[i] for i in np.sort(rng.choice(len(tr), max_train_series, replace=False))]
            r = max(0, int(getattr(ds, "_test_length", ds.windows * H)) - ds.windows * H)
            vc, vt = [], []
            for e in ds.validation_dataset:
                x = np.asarray(e["target"], dtype=np.float32)
                x = x[: len(x) - r] if r else x
                if len(x) > 2 * H:
                    vc.append(x[-max_context_store - H:-H]); vt.append(x[-H:])
            if max_train_series and len(vc) > max_train_series:
                keep = np.sort(rng.choice(len(vc), max_train_series, replace=False))
                vc, vt = [vc[i] for i in keep], [vt[i] for i in keep]
            task.train_series = tr
            task.val_contexts = vc
            task.val_targets = np.stack(vt) if vt else None
        return task

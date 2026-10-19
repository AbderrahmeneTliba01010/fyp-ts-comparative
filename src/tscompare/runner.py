"""Run one model on many tasks -> results/<track>/<model>/<task_slug>.json (resumable)."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from .metrics import QUANTILES, evaluate_forecast
from .models.base import Forecaster
from .models.baselines import SeasonalNaive
from .tasks import Task


def _gpu_peak_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2**20
    except Exception:
        pass
    return None


def _gpu_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    except Exception:
        pass


def run_one(model: Forecaster, task: Task, out_dir: Path, save_predictions=False, overwrite=False) -> dict:
    path = Path(out_dir) / model.track / model.name / f"{task.slug}.json"
    if path.exists() and not overwrite:
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(model=model.name, family=model.family, track=model.track, task=task.slug,
               freq=task.freq, term=task.term, horizon=task.horizon, domain=task.meta.get("domain"),
               contaminated=bool(task.meta.get("contaminated", False)), n_instances=task.n,
               model_info=model.info(), status="ok", error=None)
    if not model.supports(task):
        rec.update(status="unsupported")
        path.write_text(json.dumps(rec, indent=1)); return rec
    try:
        _gpu_reset()
        t0 = time.perf_counter(); model.fit(task); rec["fit_s"] = time.perf_counter() - t0
        _gpu_reset()
        t0 = time.perf_counter(); fc = model.predict(task); rec["predict_s"] = time.perf_counter() - t0
        rec["gpu_peak_mb"] = _gpu_peak_mb()
        point = np.asarray(fc.point, dtype=float)
        q = None if fc.quantiles is None else np.asarray(fc.quantiles, dtype=float)
        bad = ~np.isfinite(point).all(axis=1)
        if q is not None:
            bad |= ~np.isfinite(q).all(axis=(1, 2))
        rec["n_nonfinite"] = int(bad.sum())
        if bad.any():   # fall back to seasonal naive on failed instances (and say so)
            fb = SeasonalNaive().predict(task)
            point[bad] = fb.point[bad]
            if q is not None:
                q[bad] = fb.quantiles[bad]
        rec.update(evaluate_forecast(task, point, q))
        if save_predictions:
            np.savez_compressed(path.with_suffix(".npz"), point=point.astype(np.float32),
                                quantiles=None if q is None else q.astype(np.float32))
    except Exception as e:  # noqa: BLE001
        rec.update(status="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
    path.write_text(json.dumps(rec, indent=1, default=float))
    return rec


def run(model: Forecaster, tasks, out_dir="results", save_predictions=False, overwrite=False, verbose=True):
    recs = []
    for t in tasks:
        r = run_one(model, t, Path(out_dir), save_predictions, overwrite)
        recs.append(r)
        if verbose:
            msg = f"[{model.name:>16}] {t.slug:<40} {r['status']:<11}"
            if r["status"] == "ok":
                msg += f" MASE={r['mase']:.3f} WQL={r['wql']:.3f} fit={r.get('fit_s', 0):.1f}s pred={r.get('predict_s', 0):.1f}s"
            elif r["status"] == "failed":
                msg += f" {r['error'][:80]}"
            print(msg)
    return recs

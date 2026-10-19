#!/usr/bin/env python
"""Day-1 sanity check: instantiate + run every wrapper on a tiny synthetic task and report OK/FAIL.
Fix signature drift in the single failing class (see docs/MODEL_NOTES.md), then move on."""
import sys, time, traceback
from tscompare.models.registry import ALL_MODELS, build
from tscompare.runner import run_one
from tscompare.tasks import make_synthetic_task
from pathlib import Path
import tempfile

only = sys.argv[1:] or ALL_MODELS
task = make_synthetic_task(n_series=6, length=700, horizon=24, season=24, n_windows=1)
tmp = Path(tempfile.mkdtemp()); rows = []
for n in only:
    t0 = time.time()
    try:
        kw = dict(max_steps=30, eval_every=10) if n in ("gpt4ts", "time_llm", "tan_wo_llm", "tan_llm2attn", "tan_llm2trsf", "gpt2_scratch") else {}
        if n in ("informer", "autoformer", "fedformer", "patchtst"): kw = dict(max_steps=30)
        if n == "llmtime": kw = dict(num_samples=2)
        r = run_one(build(n, context_length=512, **kw), task, tmp, overwrite=True)
        ok = r["status"] == "ok"
        rows.append((n, "OK" if ok else r["status"].upper(), f"MASE={r.get('mase', float('nan')):.3f}" if ok else (r.get("error") or "")[:110]))
    except Exception as e:
        rows.append((n, "CRASH", f"{type(e).__name__}: {e}"[:110]))
    print(f"{rows[-1][0]:<18}{rows[-1][1]:<12}{rows[-1][2]}   ({time.time()-t0:.0f}s)", flush=True)

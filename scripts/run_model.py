#!/usr/bin/env python
"""Usage:
  python scripts/run_model.py --tier smoke --models seasonal_naive naive
  python scripts/run_model.py --tier core  --models chronos_t5_small timesfm sundial --save-predictions
  python scripts/run_model.py --tier core  --models informer patchtst gpt4ts --seeds 0 1 2
Results are resumable: finished (model, task) pairs are skipped unless --overwrite."""
import argparse, os, sys
from pathlib import Path
import yaml

from tscompare import runner
from tscompare.models.registry import ALL_MODELS, build
from tscompare.tasks import TimeBenchSource, make_synthetic_task

ap = argparse.ArgumentParser()
ap.add_argument("--tier", default="smoke", choices=["smoke", "core", "full"])
ap.add_argument("--models", nargs="+", default=["seasonal_naive"])
ap.add_argument("--protocol", default="configs/protocol.yaml")
ap.add_argument("--tasks-file", default="configs/tasks_core.yaml")
ap.add_argument("--out", default="results")
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--context-length", type=int, default=None)
ap.add_argument("--save-predictions", action="store_true")
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()

P = yaml.safe_load(open(a.protocol)); caps = P["data_caps"]; ctx = a.context_length or P["context_length"]
ov = P.get("per_model_overrides", {})

def load_tasks(need_train, max_inst):
    if a.tier == "smoke":
        return [make_synthetic_task(name=f"synthetic/{i}", seed=i, n_series=12) for i in range(4)]
    src = TimeBenchSource(os.environ["TIME_STORAGE"])
    if a.tier == "core":
        spec = yaml.safe_load(open(a.tasks_file))["tasks"]
    else:
        spec = src.list_tasks()[["name", "term"]].to_dict("records")
    out = []
    for s in spec:
        try:
            out.append(src.load(s["name"], s["term"], max_instances=max_inst, max_train_series=caps["max_train_series"],
                                max_context_store=caps["max_context_store"], seed=caps["seed"], need_train=need_train))
        except Exception as e:
            print(f"!! could not load {s}: {e}", file=sys.stderr)
    return out

for name in a.models:
    o = ov.get(name, {}); kw = {k: v for k, v in o.items() if k not in ("max_instances_per_task",)}
    sup = name in ("informer", "autoformer", "fedformer", "patchtst", "gpt4ts", "time_llm", "tan_wo_llm", "tan_llm2attn", "tan_llm2trsf", "gpt2_scratch")
    if sup:
        b = P["supervised_budget"]; kw.setdefault("max_steps", b["max_steps"])
        if name not in ("informer", "autoformer", "fedformer", "patchtst"):
            kw.update(eval_every=b["eval_every"], patience=b["patience"])
    tasks = load_tasks(sup, o.get("max_instances_per_task", caps["max_instances_per_task"]))
    for seed in (a.seeds if sup else [0]):
        m = build(name, context_length=ctx, **({"seed": seed, **kw} if sup else kw))
        if sup and len(a.seeds) > 1:
            m.name = f"{name}@s{seed}"          # aggregate seeds later with aggregate.average_seeds
        runner.run(m, tasks, out_dir=a.out, save_predictions=a.save_predictions, overwrite=a.overwrite)

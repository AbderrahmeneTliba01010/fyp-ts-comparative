"""Name -> Forecaster factory. Keep checkpoints/versions here (and in configs/models.yaml) so the paper can cite them."""
from __future__ import annotations

from .baselines import Drift, Naive, SeasonalAvg, SeasonalNaive


def build(name: str, **kw):
    ctx = kw.pop("context_length", 512)
    if name == "seasonal_naive": return SeasonalNaive(context_length=ctx)
    if name == "naive": return Naive(context_length=ctx)
    if name == "seasonal_avg": return SeasonalAvg(context_length=ctx)
    if name == "drift": return Drift(context_length=ctx)
    # ---- zero-shot
    from . import zeroshot as z
    zs = {
        "chronos_t5_small": lambda: z.Chronos("amazon/chronos-t5-small", name="chronos_t5_small", context_length=ctx, **kw),
        "chronos_t5_base": lambda: z.Chronos("amazon/chronos-t5-base", name="chronos_t5_base", context_length=ctx, **kw),
        "chronos_bolt_base": lambda: z.Chronos("amazon/chronos-bolt-base", name="chronos_bolt_base", context_length=ctx, **kw),
        "timesfm": lambda: z.TimesFM(context_length=ctx, **kw),
        "moirai": lambda: z.Moirai(context_length=ctx, **kw),
        "moment_zs": lambda: z.MomentMaskForecast(context_length=ctx, **kw),
        "sundial": lambda: z.Sundial(context_length=ctx, **kw),
        "llmtime": lambda: z.LLMTime(context_length=ctx, **kw),
    }
    if name in zs: return zs[name]()
    # ---- supervised
    if name in ("informer", "autoformer", "fedformer", "patchtst"):
        from .supervised_nf import NFModel
        return NFModel(name, context_length=ctx, **kw)
    from .patch_llm import TorchSupervised
    sv = {
        "gpt4ts":          dict(kind="gpt2", family="llm-reuse"),
        "time_llm":        dict(kind="timellm", family="llm-reuse", lr=1e-4, batch_size=32),
        "tan_wo_llm":      dict(kind="none", family="critical"),
        "tan_llm2attn":    dict(kind="attn", family="critical"),
        "tan_llm2trsf":    dict(kind="trsf", family="critical"),
        "gpt2_scratch":    dict(kind="gpt2_scratch", family="critical"),
    }
    if name in sv:
        a = {**sv[name], **kw}
        return TorchSupervised(name=name, context_length=ctx, **a)
    raise KeyError(name)


ALL_MODELS = ["seasonal_naive", "naive", "seasonal_avg", "drift", "informer", "autoformer", "fedformer", "patchtst", "llmtime", "gpt4ts", "time_llm",
              "chronos_t5_small", "timesfm", "moirai", "moment_zs", "sundial",
              "tan_wo_llm", "tan_llm2attn", "tan_llm2trsf", "gpt2_scratch"]

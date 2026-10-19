"""Zero-shot foundation-model wrappers: Chronos, TimesFM, Moirai, MOMENT, Sundial, LLMTime.

!! The upstream APIs of these libraries change quickly. Each wrapper was written against the API documented
!! in the library's README (see `docs/MODEL_NOTES.md`); run `python scripts/check_models.py` on your GPU first
!! and fix any signature drift in the *one* class that fails. Imports are lazy so the harness works without them.
"""
from __future__ import annotations

import numpy as np

from ..metrics import QUANTILES
from ..tasks import Task, pandas_freq
from .base import Forecast, Forecaster, clean_context, samples_to_forecast
from .llmtime_codec import decode, encode, fit_scaler


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _batches(n, bs):
    for i in range(0, n, bs):
        yield slice(i, min(n, i + bs))


# ------------------------------------------------------------------------------------------ Chronos
class Chronos(Forecaster):
    """amazon/chronos-t5-* (quantised tokens) or amazon/chronos-bolt-* via `chronos-forecasting`."""
    family, track, needs_gpu = "native-tsfm", "zeroshot", True

    def __init__(self, checkpoint="amazon/chronos-t5-small", name=None, batch_size=32, num_samples=20, **kw):
        super().__init__(**kw); self.ckpt, self.bs, self.ns = checkpoint, batch_size, num_samples
        self.name = name or checkpoint.split("/")[-1].replace("-", "_"); self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from chronos import BaseChronosPipeline
            self._pipe = BaseChronosPipeline.from_pretrained(self.ckpt, device_map=_device(), torch_dtype=torch.bfloat16)

    def predict(self, task: Task) -> Forecast:
        import torch
        self._load(); H = task.horizon; Q = []
        for sl in _batches(task.n, self.bs):
            ctx = [torch.tensor(clean_context(c, self.context_length)) for c in task.contexts[sl]]
            kwargs = dict(num_samples=self.ns) if "t5" in self.ckpt else {}
            q, _ = self._pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=list(QUANTILES), **kwargs)
            Q.append(q.float().cpu().numpy().transpose(0, 2, 1))        # (b,H,Q) -> (b,Q,H)
        Q = np.concatenate(Q); return Forecast(Q[:, QUANTILES.index(0.5)], Q)


# ------------------------------------------------------------------------------------------ TimesFM
class TimesFM(Forecaster):
    """google/timesfm-2.5-200m-pytorch (decoder-only, patch-based). API per the TimesFM 2.5 README."""
    name, family, track, needs_gpu = "timesfm", "native-tsfm", "zeroshot", True

    def __init__(self, checkpoint="google/timesfm-2.5-200m-pytorch", **kw):
        super().__init__(**kw); self.ckpt = checkpoint; self._m = None; self._H = None

    def _load(self, H):
        if self._m is None or self._H != H:
            import timesfm
            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.ckpt)
            m.compile(timesfm.ForecastConfig(max_context=self.context_length, max_horizon=max(H, 128),
                                             normalize_inputs=True, use_continuous_quantile_head=True,
                                             force_flip_invariance=True, infer_is_positive=False,
                                             fix_quantile_crossing=True))
            self._m, self._H = m, H

    def predict(self, task: Task) -> Forecast:
        H = task.horizon; self._load(H)
        inputs = [clean_context(c, self.context_length) for c in task.contexts]
        P, Q = [], []
        for sl in _batches(len(inputs), 256):
            point, quant = self._m.forecast(horizon=H, inputs=inputs[sl])   # quant: (b,H,10) = [mean, q0.1..q0.9]
            P.append(np.asarray(point)); Q.append(np.asarray(quant)[:, :, 1:10].transpose(0, 2, 1))
        Q = np.concatenate(Q); return Forecast(Q[:, QUANTILES.index(0.5)], Q)


# ------------------------------------------------------------------------------------------ Moirai
class Moirai(Forecaster):
    """Salesforce/moirai-1.1-R-* via uni2ts' documented GluonTS predictor path (univariate, 100 samples)."""
    name, family, track, needs_gpu = "moirai", "native-tsfm", "zeroshot", True

    def __init__(self, checkpoint="Salesforce/moirai-1.1-R-base", num_samples=100, batch_size=32, **kw):
        super().__init__(**kw); self.ckpt, self.ns, self.bs = checkpoint, num_samples, batch_size

    def predict(self, task: Task) -> Forecast:
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        H, f = task.horizon, pandas_freq(task.freq)
        model = MoiraiForecast(module=MoiraiModule.from_pretrained(self.ckpt), prediction_length=H,
                               context_length=self.context_length, patch_size="auto", num_samples=self.ns,
                               target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        ds = ListDataset([{"target": clean_context(c, self.context_length), "start": pd.Period("2000-01-01", freq=f)}
                          for c in task.contexts], freq=f)
        preds = model.create_predictor(batch_size=self.bs).predict(ds)
        S = np.stack([p.samples for p in preds])                          # (n, S, H)
        return samples_to_forecast(S)


# ------------------------------------------------------------------------------------------ MOMENT
class MomentMaskForecast(Forecaster):
    """MOMENT-1 zero-shot forecasting through its *pretraining* objective: append H masked steps to the
    (512-H)-long context and read the reconstruction. Only supports H <= 256. Point forecast only."""
    name, family, track, needs_gpu = "moment_zs", "native-tsfm", "zeroshot", True
    SEQ = 512

    def __init__(self, checkpoint="AutonLab/MOMENT-1-large", batch_size=32, **kw):
        super().__init__(**kw); self.ckpt, self.bs = checkpoint, batch_size; self._m = None

    def supports(self, task): return task.horizon <= self.SEQ // 2

    def predict(self, task: Task) -> Forecast:
        import torch
        from momentfm import MOMENTPipeline
        if self._m is None:
            self._m = MOMENTPipeline.from_pretrained(self.ckpt, model_kwargs={"task_name": "reconstruction"})
            self._m.init(); self._m.to(_device()).eval()
        H, L = task.horizon, self.SEQ - task.horizon; P = []
        for sl in _batches(task.n, self.bs):
            x = np.zeros((len(task.contexts[sl]), self.SEQ), np.float32); obs = np.zeros_like(x)
            for i, c in enumerate(task.contexts[sl]):
                c = clean_context(c, L); mu, sd = c.mean(), c.std() + 1e-6
                x[i, L - len(c):L] = (c - mu) / sd; obs[i, L - len(c):L] = 1   # left-pad zeros are unobserved
            xt, ot = torch.tensor(x[:, None]).to(_device()), torch.tensor(obs).to(_device())
            with torch.no_grad():
                out = self._m(x_enc=xt, input_mask=ot)               # masked tail positions have input_mask=0
            rec = out.reconstruction[:, 0, L:].cpu().numpy()
            for i, c in enumerate(task.contexts[sl]):
                c = clean_context(c, L); rec[i] = rec[i] * (c.std() + 1e-6) + c.mean()
            P.append(rec)
        return Forecast(np.concatenate(P), None)


# ------------------------------------------------------------------------------------------ Sundial
class Sundial(Forecaster):
    """thuml/sundial-base-128m: continuous patch tokens + flow-matching head; generates samples."""
    name, family, track, needs_gpu = "sundial", "native-tsfm", "zeroshot", True

    def __init__(self, checkpoint="thuml/sundial-base-128m", num_samples=20, batch_size=64, **kw):
        super().__init__(**kw); self.ckpt, self.ns, self.bs = checkpoint, num_samples, batch_size; self._m = None

    def predict(self, task: Task) -> Forecast:
        import torch
        from transformers import AutoModelForCausalLM
        if self._m is None:
            self._m = AutoModelForCausalLM.from_pretrained(self.ckpt, trust_remote_code=True).to(_device()).eval()
        H = task.horizon; S = []
        for sl in _batches(task.n, self.bs):
            ctx = [clean_context(c, self.context_length) for c in task.contexts[sl]]
            Lm = min(len(c) for c in ctx)       # Sundial's HF generate expects a rectangular batch -> group by length
            x = torch.tensor(np.stack([c[-Lm:] for c in ctx])).to(_device())
            mu, sd = x.mean(-1, keepdim=True), x.std(-1, keepdim=True) + 1e-6
            with torch.no_grad():
                y = self._m.generate((x - mu) / sd, max_new_tokens=H, num_samples=self.ns)   # (b, S, H)
            S.append((y * sd[:, None] + mu[:, None]).float().cpu().numpy())
        return samples_to_forecast(np.concatenate(S))


# ------------------------------------------------------------------------------------------ LLMTime
class LLMTime(Forecaster):
    """Numbers-as-text zero-shot forecasting with an *open* causal LM (original paper used GPT-3 / LLaMA-2).
    Cost grows with H and samples -> run on the 'short' tasks / capped instances (see configs/protocol.yaml)."""
    name, family, track, needs_gpu = "llmtime", "llm-reuse", "zeroshot", True

    def __init__(self, checkpoint="Qwen/Qwen2.5-1.5B", num_samples=10, prec=3, max_tokens=1536,
                 temperature=1.0, top_p=0.9, max_horizon=96, **kw):
        super().__init__(**kw); self.ckpt, self.ns, self.prec, self.mt = checkpoint, num_samples, prec, max_tokens
        self.temp, self.top_p, self.max_h = temperature, top_p, max_horizon; self._tok = self._m = None

    def supports(self, task): return task.horizon <= self.max_h

    def _load(self):
        if self._m is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.ckpt)
            self._m = AutoModelForCausalLM.from_pretrained(self.ckpt, torch_dtype=torch.bfloat16).to(_device()).eval()
            ids = set()
            for ch in "0123456789 ,":
                ids.update(self._tok.encode(ch, add_special_tokens=False)); ids.update(self._tok.encode(" " + ch, add_special_tokens=False))
            self._allowed = sorted(ids)

    def predict(self, task: Task) -> Forecast:
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList
        self._load(); H = task.horizon; allowed = torch.tensor(self._allowed)

        class Restrict(LogitsProcessor):
            def __call__(s, input_ids, scores):
                m = torch.full_like(scores, float("-inf")); m[:, allowed.to(scores.device)] = 0; return scores + m

        S = []
        for c in task.contexts:
            c = clean_context(c, self.context_length); min_, q = fit_scaler(c)
            text, width = encode(c, min_, q, self.prec)
            # shrink context until the prompt fits the token budget (leave room for H generated steps)
            tok_per_step = max(1, len(self._tok.encode(text)) // len(c)); keep = max(8, (self.mt // tok_per_step) - H)
            if len(c) > keep:
                text, width = encode(c[-keep:], min_, q, self.prec, width)
            ids = self._tok(text + " ,", return_tensors="pt").input_ids.to(_device())
            with torch.no_grad():
                gen = self._m.generate(ids, do_sample=True, temperature=self.temp, top_p=self.top_p, num_return_sequences=self.ns,
                                       max_new_tokens=H * (tok_per_step + 1), logits_processor=LogitsProcessorList([Restrict()]),
                                       pad_token_id=self._tok.eos_token_id)
            outs = self._tok.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)
            samp = np.stack([decode(o, min_, q, self.prec, width, H) for o in outs])    # (ns, H)
            med = np.nanmedian(samp, 0); samp = np.where(np.isfinite(samp), samp, med[None])
            S.append(samp)
        return samples_to_forecast(np.stack(S))

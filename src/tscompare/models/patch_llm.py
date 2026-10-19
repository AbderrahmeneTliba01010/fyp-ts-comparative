"""Compact PyTorch re-implementations (supervised, per-task training) of:

  GPT4TS / One-Fits-All  : patch -> linear -> frozen GPT-2 (6 layers; only LayerNorm + pos-emb trained) -> flatten head
  Time-LLM               : patch -> reprogramming (cross-attn to word-embedding prototypes) + statistics prompt
                           prefix -> frozen GPT-2 -> head on the patch positions
  Tan et al. ablations   : same patch+head pipeline with the LLM replaced by  none | 1 attention layer | 1 transformer layer
  gpt2_scratch           : GPT4TS recipe with RANDOMLY initialised GPT-2 (isolates the value of pretrained weights)

Deviations from the original repos (all listed in docs/PROTOCOL.md): GPT-2 backbone for Time-LLM (paper default is
LLaMA-7B), channel-independent univariate instances, per-instance normalisation, L1 loss, capped training steps.
UNTESTED on GPU in the authoring sandbox -> run scripts/check_models.py first.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from ..tasks import Task
from .base import Forecast, Forecaster, clean_context

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # keep the package importable without torch
    torch = nn = F = None


def n_patches(L, P, S):
    return (L + S - P) // S + 1


if nn is not None:

    class PatchEmbed(nn.Module):
        def __init__(self, P, S, d):
            super().__init__(); self.P, self.S = P, S; self.proj = nn.Linear(P, d)

        def forward(self, x):                                     # (B,L) -> (B,N,d)
            x = F.pad(x[:, None], (0, self.S), mode="replicate")[:, 0]
            return self.proj(x.unfold(-1, self.P, self.S))

    class Backbone(nn.Module):
        def __init__(self, kind, d, n_layers=6, n_heads=8):
            super().__init__(); self.kind = kind
            if kind == "none":
                self.m = nn.Identity()
            elif kind == "attn":
                self.m = nn.MultiheadAttention(d, n_heads, batch_first=True)
            elif kind == "trsf":
                self.m = nn.TransformerEncoderLayer(d, n_heads, 4 * d, 0.1, batch_first=True, norm_first=True)
            elif kind in ("gpt2", "gpt2_scratch"):
                from transformers import GPT2Config, GPT2Model
                g = GPT2Model.from_pretrained("gpt2") if kind == "gpt2" else GPT2Model(GPT2Config())
                g.h = g.h[:n_layers]; g.config.n_layer = n_layers
                for n, p in g.named_parameters():                 # OFA recipe: train LayerNorm + positional emb only
                    p.requires_grad = ("ln" in n) or ("wpe" in n)
                self.m = g
            else:
                raise ValueError(kind)

        def forward(self, x):
            if self.kind == "attn":
                return self.m(x, x, x, need_weights=False)[0]
            if self.kind.startswith("gpt2"):
                return self.m(inputs_embeds=x).last_hidden_state
            return self.m(x)

    class PatchModel(nn.Module):
        """patch embed -> backbone -> flatten -> linear head. Input/outputs are in instance-normalised space."""
        def __init__(self, L, H, kind, d=768, P=16, S=8, n_layers=6, dropout=0.1):
            super().__init__()
            self.emb, self.bb = PatchEmbed(P, S, d), Backbone(kind, d, n_layers)
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(n_patches(L, P, S) * d, H))

        def forward(self, xn):
            return self.head(self.bb(self.emb(xn)))

    class TimeLLMModel(nn.Module):
        def __init__(self, L, H, desc="time series", n_layers=6, d_model=32, d_ff=128, n_heads=8, P=16, S=8,
                     num_tokens=1000, prompt_len=160, dropout=0.1):
            super().__init__()
            from transformers import GPT2Model, GPT2Tokenizer
            self.llm = GPT2Model.from_pretrained("gpt2"); self.llm.h = self.llm.h[:n_layers]; self.llm.config.n_layer = n_layers
            for p in self.llm.parameters():
                p.requires_grad = False
            self.tok = GPT2Tokenizer.from_pretrained("gpt2"); self.tok.pad_token = self.tok.eos_token
            V, D = self.llm.get_input_embeddings().weight.shape
            self.mapping = nn.Linear(V, num_tokens)
            self.emb = PatchEmbed(P, S, d_model)
            self.h, self.dk = n_heads, d_model // n_heads
            self.q, self.k = nn.Linear(d_model, self.dk * n_heads), nn.Linear(D, self.dk * n_heads)
            self.v, self.o = nn.Linear(D, self.dk * n_heads), nn.Linear(self.dk * n_heads, D)
            self.N, self.d_ff, self.H, self.L, self.desc, self.plen = n_patches(L, P, S), d_ff, H, L, desc, prompt_len
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(d_ff * self.N, H))

        def reprogram(self, pt):
            B, N, _ = pt.shape
            src = self.mapping(self.llm.get_input_embeddings().weight.t()).t()          # (num_tokens, D)
            q = self.q(pt).view(B, N, self.h, self.dk)
            k, v = self.k(src).view(-1, self.h, self.dk), self.v(src).view(-1, self.h, self.dk)
            a = torch.softmax(torch.einsum("bnhe,she->bhns", q, k) / math.sqrt(self.dk), -1)
            return self.o(torch.einsum("bhns,she->bnhe", a, v).reshape(B, N, -1))

        @torch.no_grad()
        def make_prompts(self, xn):
            x = xn.float(); f = torch.fft.rfft(x, dim=-1)
            ac = torch.fft.irfft(f * torch.conj(f), n=x.shape[-1], dim=-1); ac[:, 0] = float("-inf")
            lags = ac.topk(5, dim=-1).indices.cpu().tolist()
            mn, mx, md = x.min(-1).values.tolist(), x.max(-1).values.tolist(), x.median(-1).values.tolist()
            up = (x[:, -1] > x[:, 0]).tolist()
            return [f"Dataset description: {self.desc}; Task description: forecast the next {self.H} steps "
                    f"given the previous {self.L} steps; Input statistics: min value {mn[i]:.3f}, max value {mx[i]:.3f}, "
                    f"median value {md[i]:.3f}, the trend of input is {'upward' if up[i] else 'downward'}, "
                    f"top 5 lags are : {lags[i]}" for i in range(x.shape[0])]

        def forward(self, xn):
            ids = self.tok(self.make_prompts(xn), return_tensors="pt", padding="max_length", max_length=self.plen,
                           truncation=True).input_ids.to(xn.device)
            x = torch.cat([self.llm.get_input_embeddings()(ids), self.reprogram(self.emb(xn))], 1)
            h = self.llm(inputs_embeds=x).last_hidden_state[:, -self.N:, :self.d_ff]
            return self.head(h)


def make_windows(series, L, H, n, rng, min_ctx=16):
    """Random (context, target) windows from training series; short contexts are left-padded with the first value."""
    ok = [s for s in series if len(s) >= H + min_ctx]
    if not ok:
        raise ValueError("no training series long enough for this horizon")
    X, Y = np.empty((n, L), np.float32), np.empty((n, H), np.float32)
    for i in range(n):
        s = ok[rng.integers(len(ok))]; e = int(rng.integers(min_ctx, len(s) - H + 1))
        c = s[max(0, e - L):e]
        X[i] = np.concatenate([np.full(L - len(c), c[0], np.float32), c]); Y[i] = s[e:e + H]
    return X, Y


def pad_to(c, L):
    c = clean_context(c, L)
    return c if len(c) == L else np.concatenate([np.full(L - len(c), c[0], np.float32), c])


class TorchSupervised(Forecaster):
    track, needs_gpu = "supervised", True

    def __init__(self, kind, name, family, lr=5e-4, batch_size=128, max_steps=1500, eval_every=100, patience=4,
                 max_windows=40000, seed=0, **model_kw):
        if torch is None:
            raise ImportError("PyTorch (+ transformers for gpt2 variants) is required for this model")
        super().__init__(**{k: v for k, v in model_kw.items() if k == "context_length"})
        self.kind, self.name, self.family = kind, name, family
        self.lr, self.bs, self.max_steps, self.eval_every, self.patience = lr, batch_size, max_steps, eval_every, patience
        self.max_windows, self.seed = max_windows, seed
        self.mkw = {k: v for k, v in model_kw.items() if k != "context_length"}; self.model = None

    def _make(self, L, H, task):
        if self.kind == "timellm":
            return TimeLLMModel(L, H, desc=f"{task.name} ({task.freq})", **self.mkw)
        return PatchModel(L, H, self.kind, **self.mkw)

    @staticmethod
    def _norm(x):
        mu = x.mean(1, keepdim=True); sd = x.std(1, keepdim=True, unbiased=False) + 1e-5
        return (x - mu) / sd, mu, sd

    def fit(self, task: Task):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        L, H = self.context_length, task.horizon
        tr = [clean_context(s) for s in task.train_series]
        X, Y = make_windows(tr, L, H, self.max_windows, rng)
        if task.val_contexts:
            vX = np.stack([pad_to(c, L) for c in task.val_contexts]); vY = np.nan_to_num(task.val_targets.astype(np.float32))
        else:
            vX, vY = make_windows(tr, L, H, 1000, np.random.default_rng(self.seed + 1))
        vX, vY = vX[:2000], vY[:2000]
        self.L, self.model = L, self._make(L, H, task).to(dev)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.max_steps)
        amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda"))
        tnames = {n for n, p in self.model.named_parameters() if p.requires_grad}
        best, bad, best_state = float("inf"), 0, None

        def loss_on(xb, yb, train):
            xb, yb = torch.tensor(xb, device=dev), torch.tensor(yb, device=dev)
            xn, mu, sd = self._norm(xb)
            with torch.autocast(**amp):
                out = self.model(xn)
            return F.l1_loss(out.float(), (yb - mu) / sd)

        for step in range(1, self.max_steps + 1):
            self.model.train(); idx = rng.integers(0, len(X), self.bs)
            loss = loss_on(X[idx], Y[idx], True); opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            if step % self.eval_every == 0:
                self.model.eval()
                with torch.no_grad():
                    v = float(np.mean([loss_on(vX[i:i + 256], vY[i:i + 256], False).item() for i in range(0, len(vX), 256)]))
                if v < best - 1e-5:
                    best, bad = v, 0
                    best_state = {k: t.detach().clone() for k, t in self.model.state_dict().items() if k in tnames}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state:
            self.model.load_state_dict(best_state, strict=False)
        self.val_l1 = best

    def predict(self, task: Task) -> Forecast:
        dev = "cuda" if torch.cuda.is_available() else "cpu"; self.model.eval(); P = []
        amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda"))
        for i in range(0, task.n, 256):
            x = torch.tensor(np.stack([pad_to(c, self.L) for c in task.contexts[i:i + 256]]), device=dev)
            xn, mu, sd = self._norm(x)
            with torch.no_grad(), torch.autocast(**amp):
                o = self.model(xn)
            P.append((o.float() * sd + mu).cpu().numpy())
        return Forecast(np.concatenate(P), None)

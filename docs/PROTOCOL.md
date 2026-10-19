# Experimental protocol

*This file is the contract. Anything not written here must not differ between methods.*

## Research questions
| | Question | Evidence produced |
|---|---|---|
| RQ1 | Under a leakage-controlled benchmark, which family wins: pre-LLM Transformers, LLM-reuse, or natively pretrained TSFMs — and in which regimes (domain, frequency, horizon)? | main table, CD diagram, group heatmaps |
| RQ2 | Do pretrained LLM *weights* help? (Tan et al. 2024 replicated on fresh data) | GPT4TS / Time-LLM vs `tan_*` ablations vs `gpt2_scratch` |
| RQ3 | What does accuracy cost (training + inference time, memory)? | accuracy-vs-cost plot |
| RQ4 | Do methods fail disproportionately around sharp local dynamics? (motivates our micro-patching method) | `analysis/mechanism.py`, patch-size sweep |
| RQ5 | How sensitive are rankings to context length and to contamination filtering? | ctx 96/336/512/2048; with/without flagged pairs |

## Benchmarks
* **Primary — TIME** (Qiao et al., ICML 2026): fresh datasets designed to avoid leakage; 98 tasks; short/medium/long terms. Official `timebench` package supplies windows, horizons, splits → identical to the public leaderboard.
* **Optional robustness — fev-bench** (Shchur et al., 2025) and **GIFT-Eval** (Aksu et al., 2024). GIFT-Eval overlaps with several pretraining corpora → use only its leakage-free subset, flagged in the contamination ledger. Agreement of rankings across benchmarks is itself a result.
* Why not ETT/Weather/ECL/Traffic: the original papers of these methods report on them, they are heavily reused in pretraining corpora, and their test sets are the most over-tuned in the field. We still *may* add them in an appendix purely to compare against numbers published in the original papers.

## Two regimes, one test set
| Regime | Methods | What they see |
|---|---|---|
| zero-shot | LLMTime, Chronos, TimesFM, Moirai, MOMENT (mask-forecast), Sundial, baselines | context window only |
| supervised (per-task training) | Informer, Autoformer, FEDformer, PatchTST, GPT4TS, Time-LLM, Tan variants, `gpt2_scratch` | `train_series` + one validation window per series, then context window |

Both regimes are scored on **exactly the same test windows** and appear in one table (column `track`). Supervised methods are trained from scratch **per task** with an identical budget (`protocol.yaml → supervised_budget`). Foundation models are *not* fine-tuned in the main table (fine-tuned variants may be an extension).

## Fairness rules
1. Same instances: univariate / channel-independent, fixed seed subsample (`max_instances_per_task`), identical for every model.
2. Same context cap (512) in the main table; native-context behaviour is studied separately (RQ5).
3. Same point forecast for point metrics: the **median** (q=0.5) if the model is probabilistic.
4. Same loss for everything we train: L1 on instance-normalised data. Same step cap / early stopping / validation window.
5. No test-time information: test targets never reach `fit`; hyper-parameters are fixed a priori (no per-dataset tuning for any method). If you tune anything, tune it for **all** methods on validation windows only.
6. Failures, unsupported tasks (horizon caps) and non-finite outputs are **reported** and imputed with seasonal naive (fev-bench rule); coverage is printed next to scores.
7. Report ≥3 seeds for every trained model and for sampling-based models (mean ± std); single-seed numbers are debugging only.

## Metrics
* **MASE** (headline): `mean|y−ŷ| / mean|y_t − y_{t−m}|`, scale on the full in-sample context, `m` = gluonts seasonality of the frequency.
* **WQL** (mean weighted quantile loss over τ=0.1…0.9; CRPS proxy): only probabilistic models; models without quantiles are *not* imputed on it.
* nMAE / nMSE (context-z-scored): for comparability with the LTSF-style numbers in the original papers.
* **Aggregation**: per-task metric → divide by seasonal naive → geometric mean over tasks (+ bootstrap 95% CI over tasks), skill score `1−GM`, average rank, pairwise win rate, Friedman + Nemenyi critical difference, paired Wilcoxon on log-ratios for targeted comparisons.

## Contamination
Pretrained models may have seen benchmark series. Maintain `configs/contamination.csv` (corpus, data cutoff, overlap with TIME sources). Report the main table with and without flagged (model, task) pairs; conclusions that flip are flagged as fragile. LLM-based methods (LLMTime, Time-LLM, GPT4TS backbones) cannot be audited at series level — state it in Limitations.

## Known deviations (copy into the paper)
* All models run channel-independent/univariate; Informer/Autoformer/FEDformer originally mix channels and use calendar embeddings (omitted).
* Time-LLM uses a GPT-2 backbone (paper: LLaMA-7B); LLMTime uses an open 1.5B LM (paper: GPT-3 / LLaMA-2) and is limited to horizon ≤ 96 and 100 instances/task for cost.
* GPT4TS/Time-LLM are compact re-implementations (`patch_llm.py`), not the authors' repos; validate against the authors' reported ETT numbers once (see `docs/PLAN.md`, step 4).
* TimesFM = v2.5 checkpoint; the deck describes the v1 design (same decoder-only patch family).
* MOMENT is evaluated through its masked-reconstruction pretraining objective (zero-shot); a linear-probe variant is an extension.
* Seasonal-naive quantiles are empirical in-context residual quantiles (our construction; point forecast matches the reference definition).

## Threats to validity (write these up honestly)
Subsampled instances; per-task training budget may under-train the large models; GPT-2 vs LLaMA-7B; contamination audit incomplete; TIME domains skew; hyper-parameter defaults favour whichever library ships better defaults (mitigate with the validation-only tuning rule above).

# Model notes (what to fix first when `check_models.py` fails)

All wrappers import lazily and were written against the libraries' public READMEs; **none of the GPU/HF paths could be executed in the authoring sandbox** (no GPU, no HF access). Expect 1–3 small signature fixes. Each lives in exactly one class.

| Method | Class (file) | Library / checkpoint | Most likely breakage |
|---|---|---|---|
| Chronos (T5/Bolt) | `Chronos` (zeroshot.py) | `chronos-forecasting`, `amazon/chronos-t5-small`, `amazon/chronos-bolt-*` | `predict_quantiles` kwargs (`num_samples` only for T5); Chronos-2 uses a different pipeline class |
| TimesFM | `TimesFM` | `timesfm` 2.5, `google/timesfm-2.5-200m-pytorch` | `ForecastConfig` fields / `forecast` return layout (we assume `[mean, q0.1..q0.9]`) |
| Moirai | `Moirai` | `uni2ts`, `Salesforce/moirai-1.1-R-base`, gluonts predictor path | gluonts/uni2ts version pins; freq alias handling |
| MOMENT | `MomentMaskForecast` | `momentfm`, `AutonLab/MOMENT-1-large` | masking arguments (`input_mask` vs `mask`); verify that masked tail is actually reconstructed |
| Sundial | `Sundial` | `thuml/sundial-base-128m` (`trust_remote_code`) | `generate` signature; rectangular batches |
| LLMTime | `LLMTime` + `llmtime_codec.py` | any HF causal LM (default Qwen2.5-1.5B) | tokenizer digit splitting: check `tok_per_step`; codec itself is unit-tested |
| Informer/Autoformer/FEDformer/PatchTST | `NFModel` (supervised_nf.py) | `neuralforecast` | model kwargs per version; FEDformer needs `input_size`/`h` compatible with its modes |
| GPT4TS, Time-LLM, Tan variants, gpt2_scratch | `TorchSupervised`, `PatchModel`, `TimeLLMModel` (patch_llm.py) | `torch`, `transformers` (`gpt2`) | HF GPT-2 internals when slicing `h`; memory with patch 4/2 sweeps |

## Verification you should do once (cheap, catches silent bugs)
1. `seasonal_naive` MASE vs TIME's published seasonal-naive per task (notebook 01, §4).
2. Reproduce one published number per family within noise, on the *original* benchmark: e.g. PatchTST or GPT4TS on ETTh1 (96-step) vs the paper. If your re-implementation is far off, fix it before using it to make claims.
3. `tan_wo_llm` should be within ~±5% of `gpt4ts` on most tasks (Tan et al.'s finding). A huge gap either way means a bug or a genuinely new finding — check bugs first.
4. Probabilistic wrappers: empirical coverage of the 10–90% interval should be roughly 80% (add a quick check on one task).

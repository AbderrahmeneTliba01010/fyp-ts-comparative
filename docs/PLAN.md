# Plan (≈3 weeks, adjust to your GPU)

| Step | What | Done when |
|---|---|---|
| 1 | `00` notebook: install, smoke tier, `check_models.py` | every method you will report says OK |
| 2 | `01` notebook: download TIME, choose core tasks, leakage assertions, **seasonal-naive vs leaderboard** check | gap to published seasonal naive is small |
| 3 | Run zero-shot block (Chronos, TimesFM, Moirai, Sundial, MOMENT) on core tasks | `results/zeroshot/*` complete, coverage table clean |
| 4 | Validate re-implementations against the *original* papers once (one ETT number each for GPT4TS / PatchTST) | within noise |
| 5 | Supervised block, seed 0 → then seeds 0-2 | 3 seeds per model |
| 6 | LLMTime (capped), Tan ablations, `gpt2_scratch` | RQ2 table fillable |
| 7 | `03` notebook → figures/tables into `paper/` ; `04` probes | paper compiles with real tables |
| 8 | Write: results, analysis, limitations; contamination ledger completed | draft to professor |

## Rough compute guide (estimates, not measurements — calibrate on `smoke` + one real task first)
Zero-shot models: minutes per task on one 24 GB GPU. LLMTime: hours (cap instances/horizon). Per-task supervised training: ~5-30 min/task/model/seed at 1500 steps (Time-LLM slowest). Core tier (20 tasks) × 11 trained configs × 3 seeds ≈ 650 trainings → budget accordingly or cut to 12 tasks / 2 seeds and say so.

## Decisions to confirm with your professor
1. Core subset size (20 tasks?) and whether the full 98-task zero-shot run is expected.
2. TIME as primary benchmark (+ fev-bench/GIFT-Eval as robustness) vs. also reporting classic LTSF sets.
3. Zero-shot FMs not fine-tuned in the main table (fine-tuning as extension).
4. GPT-2 instead of LLaMA-7B for Time-LLM and an open LM for LLMTime (compute-driven).
5. Whether the micro-patching sweep (RQ4) belongs in this paper or is held for the method paper.

## Link to our method
RQ4 + the patch-size sweep produce the *evidence of the gap* (where existing tokenisations fail) that the method paper needs as motivation; keep the same harness so the new method is a one-class addition (`Forecaster` subclass + registry line).

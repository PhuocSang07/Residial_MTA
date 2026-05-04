# Phase 05 — Evaluation + Ablation

## Context Links

- Eval entry point: `MTA/distillm-master/evaluate_main.py`
- Rouge metric: `MTA/distillm-master/rouge_metric.py::compute_metrics`
- PromptDataset: `MTA/distillm-master/data_utils/prompt_datasets.py`
- Existing eval scripts: `MTA/distillm-master/scripts/{gpt2,qwen1.5}/sft/sft_*.sh` (have eval-only modes)
- Brainstorm spec §3.4 + §6 + §7
- Distilled checkpoints: produced by Phase 4

## Overview

- **Priority:** P1
- **Status:** pending
- **Description:** Run Rouge-L evaluation on 5 benchmarks × 4 baselines × 2 setups × 3 seeds = ~120 eval runs. Aggregate to mean ± std table. Run γ-sweep ablation on 1 setup/seed.

## Key Insights from Scouting

- `evaluate_main.py` runs generation per `args.data_names` over `PromptDataset(args.data_dir)`. Set `--data-names eval-set --data-dir data/{set}` per call.
- 5 eval datasets (Dolly subset, SelfInst, VicunaEval, S-NI, UnNI) all use the same dataloader pattern.
- Generation config: greedy decoding (`--no-do-sample`) for deterministic eval; max-length matches train.
- Each eval is fast: ~5-10 min per benchmark per checkpoint on 1 GPU.
- Total eval runs: 5 benchmarks × ~12 checkpoints (2 setups × 4 baselines × ~1.5 effective seeds after dedup) ≈ 60-120 runs ≈ 8-15 GPU-hours.

## Requirements

**Functional:**
1. Eval each distilled checkpoint on all 5 benchmarks → produce JSONL of generated responses.
2. Compute Rouge-L per (checkpoint × benchmark) using `compute_metrics`.
3. Aggregate across 3 seeds: mean ± std table.
4. Generate comparison table: rows = (SFT, Residual γ=0, SpanFDD, SpanResidual) × (GPT2 setup, Qwen setup); cols = 5 benchmarks + AVG.
5. γ-sweep ablation (1 setup, 1 seed): γ ∈ {0, 0.5, 1.0, 2.0} → reuses Phase 3 trainer with different `--gamma-span`.

**Non-functional:**
- Eval scripts mirror existing `scripts/gpt2/sft/sft_*.sh` patterns with `--do-eval` flag (no `--do-train`).
- Aggregation script in Python under `tools/` directory; ≤ 150 LOC.

## Architecture

```
For each checkpoint in results/{model}/{config}/seed{N}/:
  For each eval_set in [dolly_subset, self-inst, vicuna, sinst, uinst]:
    bash scripts/spanresidual/eval-checkpoint.sh \
        ${CKPT} ${EVAL_SET} \
      → eval_outputs/{config}_seed{N}_{eval_set}/responses.jsonl
      → eval_outputs/{config}_seed{N}_{eval_set}/rouge.json

Aggregate:
  tools/eval_aggregate.py
    → results_table.md (mean ± std per (config, benchmark))
```

## Related Code Files

**Modify:**
- None. `evaluate_main.py` already supports the eval-only mode.

**Create:**
- `MTA/distillm-master/scripts/spanresidual/eval-checkpoint.sh` — generic launcher: takes `${CKPT}`, `${EVAL_SET}`, `${DATA_DIR}`. Calls `evaluate_main.py` via torchrun. (~50 LOC.)
- `MTA/distillm-master/scripts/spanresidual/eval-all.sh` — loops over `(checkpoint × eval_set)` matrix. (~40 LOC.)
- `MTA/distillm-master/tools/eval_aggregate.py` — reads `eval_outputs/*/rouge.json`, parses (config, seed, eval_set), produces table mean ± std. Outputs Markdown table to `eval_outputs/results_table.md`. (~120 LOC.)
- `MTA/distillm-master/scripts/spanresidual/ablation-gamma-sweep.sh` — runs Phase 3 trainer 4 times with γ ∈ {0, 0.5, 1.0, 2.0}, then evals each. (~40 LOC.)

**Delete:** none.

## Implementation Steps

1. **Author `eval-checkpoint.sh`:**
   ```bash
   #!/bin/bash
   CKPT=${1}; EVAL_SET=${2}; DATA_DIR=${3}
   torchrun --nproc_per_node 1 evaluate_main.py \
     --model-path ${CKPT} \
     --data-dir ${DATA_DIR} \
     --data-names ${EVAL_SET} \
     --eval-batch-size 8 \
     --max-length 256 --max-prompt-length 128 \
     --do-eval \
     --save eval_outputs/${EVAL_SET}/$(basename ${CKPT}) \
     --eval-gen \
     --top-k 0 --top-p 1.0 --temperature 1.0 \
     --no-repeat-ngram-size 6
   ```
2. **Author `eval-all.sh`:**
   ```bash
   #!/bin/bash
   CHECKPOINT_ROOT=results
   for CKPT in $(find ${CHECKPOINT_ROOT} -name "best" -type d); do
     for EVAL in dolly self-inst vicuna sinst uinst; do
       bash scripts/spanresidual/eval-checkpoint.sh ${CKPT} ${EVAL} data/${EVAL}
     done
   done
   ```
3. **Author `tools/eval_aggregate.py`:**
   - Walk `eval_outputs/`, parse path pattern `{eval_set}/{config}_seed{N}/`.
   - Read `rouge.json` (compute_metrics output).
   - Group by (config, eval_set), compute mean + std across seeds.
   - Emit Markdown:
     ```
     | Config        | Dolly | SelfInst | Vicuna | S-NI | UnNI | AVG |
     |---------------|-------|----------|--------|------|------|-----|
     | SFT           | XX.XX±YY | ... |
     | Residual γ=0  | ...
     | SpanFDD       | ...
     | SpanResidual  | ...
     ```
   - Highlight best-per-row in bold.
4. **Run full eval matrix.** ~8-15 GPU-hours total.
5. **Run γ-sweep ablation:**
   - Author `ablation-gamma-sweep.sh`. For each γ ∈ {0, 0.5, 1.0, 2.0}, launch Phase 3 trainer with that γ on GPT2 setup, seed 42.
   - Eval each on Dolly only (faster signal).
   - Aggregate into `eval_outputs/ablation_gamma.md`.
6. **Sanity comparison vs paper.** Confirm Residual γ=0 result on Dolly Rouge-L is within ±0.5 of paper Table 1 / 2 (whichever matches our setup closest). If not, document discrepancy and root-cause (likely teacher-SFT delta or hyperparameter difference).
7. **Outlier check.** If any (config, seed) has Rouge-L > 2 std from mean, re-run that seed.

## Todo Checklist

- [ ] Write `eval-checkpoint.sh`
- [ ] Write `eval-all.sh`
- [ ] Write `tools/eval_aggregate.py`
- [ ] Write `ablation-gamma-sweep.sh`
- [ ] Run full eval matrix (5 benchmarks × all checkpoints)
- [ ] Aggregate results into Markdown table
- [ ] Run γ-sweep ablation (4 γ values, 1 setup, 1 seed)
- [ ] Sanity vs paper Residual baseline (±0.5 Rouge-L)
- [ ] Outlier re-runs if needed

## Success Criteria

- `eval_outputs/results_table.md` exists with all 4 rows × 6 columns × 2 setups populated.
- `eval_outputs/ablation_gamma.md` shows γ-sweep curve.
- **Primary:** SpanResidual avg Rouge-L > Residual γ=0 on both setups (statistically meaningful, ≥ 1× std gap).
- **Secondary:** SpanResidual ≥ SpanFDD on both setups.
- **Tertiary:** Residual γ=0 reproduces paper within ±0.5.
- **Ablation:** γ-sweep curve shows non-zero γ optimal (e.g. argmax γ ∈ {0.5, 1.0, 2.0}).

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Eval datasets missing fields expected by `PromptDataset` | M | H | Phase 1 verification step — catches mismatches before Phase 5 |
| Generation OOM (max-length 256 + larger batches) | L | M | Reduce eval batch to 4 |
| `rouge_metric.compute_metrics` differs from paper's tokenizer | M | M | Already used in `span_fdd_finetune.py::evaluate` — same metric throughout |
| Reproducibility: greedy decoding still has variance | L | L | Set `torch.manual_seed`, fix `do_sample=False` |
| Non-monotonic γ-sweep | L | L | Document as observation; expected if γ=2 over-regularizes span |
| Total compute exceeds available time | M | M | Eval is parallelizable across GPUs; partition by benchmark |

## Test Matrix

| Test | Type | Pass Criteria |
|------|------|---------------|
| Sanity: SFT eval matches existing SpanFDD repo numbers | sanity | Rouge-L within ±0.3 |
| Reproducibility: SpanResidual seed 42 vs 1337 | seed-stability | std < 1.0 |
| Comparison: SpanResidual > Residual γ=0 | primary | Mean delta > 0, ≥ 1×std |
| Comparison: SpanResidual ≥ SpanFDD | secondary | Mean delta ≥ 0 |
| Ablation: γ-sweep | ablation | argmin not at γ=0 OR argmin not at γ=2 |

## Rollback Plan

Pure read-only from checkpoints. Aggregation script is idempotent; can re-run after fixing parsing bugs without re-evaluating.

## Next Steps / Dependencies

- **Blocked by:** Phase 4 (checkpoints).
- **Unblocks:** Phase 6 (writeup).

## Unresolved Questions

- **Q3 ablation candidate:** Is "span on last layer only" worth a sweep? Add as Phase 5b stretch goal if time permits — flips `student_layer_mapping` to last-layer index only.
- Possible outcome: SpanResidual ≈ Residual γ=0. If so, hypothesis (span ⊥ residual) is invalidated. Document neutrally; add post-hoc analysis on which span layers contributed.

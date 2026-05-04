# Phase 04 — Training Scripts + Actual Runs

## Context Links

- Reference scripts: `MTA/distillm-master/scripts/gpt2/spanfdd/train_0.1B_1.5B.sh`, `scripts/qwen1.5/spanfdd/train_0.5B_1.8B.sh`
- Trainer entry: `MTA/distillm-master/span_residual_finetune.py` (Phase 3)
- Pretrained projectors: `results/{gpt2,qwen1.5}/projectors/spanresidual_*/projector_best.pt` (Phase 2)

## Overview

- **Priority:** P1
- **Status:** pending
- **Description:** Author bash launchers mirroring SpanFDD scripts; launch SpanResidual training for both setups + 4 baseline configurations needed for comparison.

## Key Insights from Scouting

- Existing `spanfdd/train_*.sh` launches via `torchrun ... span_fdd_finetune.py` with deepspeed config.
- Standard hyperparameters (gpt2): batch 16, grad-accum 1, lr 1e-4, 5 epochs, max-length 256.
- Standard hyperparameters (qwen): batch 8, grad-accum 2 (effective 16), lr 1e-4, 5 epochs.
- Layer mappings already chosen in spanfdd scripts: gpt2 `t=[16 24 32] s=[4 6 8] split=[0 1 3 3]`; qwen `t=[4 8 16] s=[4 8 16] split=[0 1 3 3]`. Reuse.
- DDP distributed sampler handles 1-2 GPU automatically.
- `--w-span-loss` arg is the existing γ in span code; we override via `--gamma-span` in Phase 3 → keep both consistent (use `args.gamma_span` for new code, leave `args.w_span_loss` in `span_utils.py` paths via legacy plumbing).

## Requirements

**Functional:**
1. Launcher scripts for SpanResidual on both setups: gpt2-1.5B→0.1B, qwen1.5-1.8B→0.5B.
2. Baseline launcher scripts for fair comparison:
   - SFT only (existing `sft_*.sh` reused — verify present).
   - Residual paper (γ=0): `train-spanresidual-gpt2-gamma0-1.5b.sh` etc.
   - SpanFDD (existing `spanfdd/train_*.sh` reused — verify present).
   - SpanResidual (γ=1.0): `train-spanresidual-gpt2-1.5b.sh` etc.
3. Each script saves to a unique `results/{model}/spanresidual/{config}_seed{N}/`.
4. Launch with seeds 42, 1337, 2024.

**Non-functional:**
- Stay within 1-2 GPU 16-40GB budget. GPT2 setup runs on 1× 16GB; Qwen on 1× 24-40GB.
- Total compute estimate: 2 setups × 4 configs × 3 seeds × ~3h = 72 GPU-hours.
- Reduce by sharing SFT-only and SpanFDD baselines from existing runs if available.

## Architecture

```
Run matrix:
              GPT2 setup            Qwen1.5 setup
SFT             [reuse]             [reuse]
Residual γ=0    seeds 42,1337,2024  seeds 42,1337,2024
SpanFDD         [reuse if exists]   [reuse if exists]
SpanResidual    seeds 42,1337,2024  seeds 42,1337,2024
```

Per script: `torchrun --nproc_per_node 1 span_residual_finetune.py {OPTS}` → produces checkpoint at `results/{model}/spanresidual/{config}_seed{N}/`.

## Related Code Files

**Modify:** none (no code changes; this phase only launches Phase 3 code).

**Create:**
- `MTA/distillm-master/scripts/gpt2/spanresidual/train-0.1b-1.5b-residual.sh` (γ=0, λ=0.5; reproduces paper Residual baseline using our Stage-2 path)
- `MTA/distillm-master/scripts/gpt2/spanresidual/train-0.1b-1.5b-spanresidual.sh` (γ=1.0, λ=0.5; our method)
- `MTA/distillm-master/scripts/qwen1.5/spanresidual/train-0.5b-1.8b-residual.sh`
- `MTA/distillm-master/scripts/qwen1.5/spanresidual/train-0.5b-1.8b-spanresidual.sh`
- `MTA/distillm-master/scripts/spanresidual/launch-all-seeds.sh` — wrapper that loops over seeds, calls the 4 above. Avoids 12 separate scripts.

**Delete:** none.

## Implementation Steps

1. **Author `train-0.1b-1.5b-spanresidual.sh`** by cloning `scripts/gpt2/spanfdd/train_0.1B_1.5B.sh` and editing:
   - Entry point: `span_residual_finetune.py` (replace `span_fdd_finetune.py`)
   - Add: `--projector-load-path ${BASE_PATH}/results/gpt2/projectors/spanresidual_1.5B/projector_best.pt`
   - Add: `--lambda-residual 0.5 --gamma-span 1.0 --d-bottleneck 64`
   - Override `--save`: `${BASE_PATH}/results/gpt2/spanresidual/spanresidual_0.1B_1.5B_seed${SEED}`
   - Drop irrelevant DistiLLM args: `--student-gen`, `--init-threshold`, `--loss-eps`, `--capacity`, `--type adaptive-srkl` (since we don't use student-gen / replay).
   - Pass through `${SEED}` env var.
2. **Author `train-0.1b-1.5b-residual.sh`** — same as above but `--gamma-span 0`. Output dir: `.../residual_0.1B_1.5B_seed${SEED}/`.
3. **Author Qwen counterparts** — clone `scripts/qwen1.5/spanfdd/train_0.5B_1.8B.sh`, apply same edits + Qwen layer mapping (already in source).
4. **Author `launch-all-seeds.sh`:**
   ```bash
   #!/bin/bash
   set -e
   for SEED in 42 1337 2024; do
     for SCRIPT in \
       scripts/gpt2/spanresidual/train-0.1b-1.5b-residual.sh \
       scripts/gpt2/spanresidual/train-0.1b-1.5b-spanresidual.sh \
       scripts/qwen1.5/spanresidual/train-0.5b-1.8b-residual.sh \
       scripts/qwen1.5/spanresidual/train-0.5b-1.8b-spanresidual.sh ; do
       SEED=${SEED} bash ${SCRIPT}
     done
   done
   ```
   Each script reads `${SEED:-42}` env var and passes via `--seed`. Save log per (script, seed) pair.
5. **Pre-flight checks before launch:**
   - `nvidia-smi` confirms target GPU available.
   - `ls results/gpt2/projectors/spanresidual_1.5B/projector_best.pt` returns non-empty.
   - `ls processed_data/dolly/full/{gpt2,qwen}/train.bin` non-empty.
6. **Launch GPT2 runs first** (~6× 3h = 18h on 1 GPU; or parallel on 2 GPUs ~9h):
   - 3 seeds × 2 configs (residual γ=0, spanresidual γ=1.0).
7. **Launch Qwen runs** (~6× 4h = 24h):
   - Same matrix.
8. **(Re-)run SFT and SpanFDD baselines** if not already in results:
   - SFT: `bash scripts/gpt2/sft/sft_base.sh` × 3 seeds.
   - SpanFDD: `bash scripts/gpt2/spanfdd/train_0.1B_1.5B.sh` × 3 seeds.
   - Same for Qwen.
9. **Monitor:** tail per-run `log.txt`; abort if any of L_SFT/L_res/L_span produces NaN within first 50 steps.

## Todo Checklist

- [ ] Write `train-0.1b-1.5b-spanresidual.sh`
- [ ] Write `train-0.1b-1.5b-residual.sh` (γ=0)
- [ ] Write `train-0.5b-1.8b-spanresidual.sh`
- [ ] Write `train-0.5b-1.8b-residual.sh`
- [ ] Write `launch-all-seeds.sh`
- [ ] Pre-flight checks
- [ ] Launch GPT2 runs (3 seeds × 2 configs)
- [ ] Launch Qwen runs (3 seeds × 2 configs)
- [ ] Run/reuse SFT baselines
- [ ] Run/reuse SpanFDD baselines
- [ ] Confirm all checkpoints saved + dev Rouge-L logged

## Success Criteria

- 12 distilled checkpoints under `results/{model}/spanresidual/` (4 configs × 3 seeds × no, sorry: 2 setups × 2 configs × 3 seeds = 12 SpanResidual+Residual checkpoints) plus SFT/SpanFDD baselines.
- Each run's `log.txt` shows monotonically decreasing avg loss across epochs.
- Dev Rouge-L logged per epoch; final-epoch dev Rouge-L > SFT baseline by ≥ 0.5 for SpanResidual.

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Long total wall-clock (~50-70 GPU-hours) | H | M | Run in parallel on 2 GPUs; reduce to 1 seed for ablation runs |
| Disk full (12+ checkpoints × ~600MB each) | M | M | Save only best-on-dev checkpoint; delete intermediate optimizer states |
| Phase 3 bug surfaces only mid-training | M | H | 50-iter smoke from Phase 3 caught most; monitor first 200 steps closely |
| OOM on Qwen with batch 8 + grad-accum 2 + spans | M | M | Reduce max-length 256→192; or batch 4 grad-accum 4 |
| Reproducibility variance > 1.0 Rouge-L across seeds | M | M | 3 seeds gives mean ± std; if std > 1.0, add 2 more seeds for top configs |
| Different DeepSpeed checkpointing format breaks loading P_T→A | M | H | Tested in Phase 3 smoke; document workaround in trainer (load AFTER deepspeed.initialize) |

## Compute Budget Detail

| Setup | Effective batch | Steps/epoch | Epochs | Steps total | Time/step | Per-run | × 6 (3 seeds × 2 cfg) |
|-------|----------------|-------------|--------|-------------|-----------|---------|-----------------------|
| GPT2-0.1B  | 16 | ~700 | 5 | 3500 | 1.5s | ~1.5h | ~9h |
| Qwen-0.5B  | 16 | ~700 | 5 | 3500 | 3.0s | ~3h | ~18h |

Total: ~27 GPU-hours for SpanResidual + Residual (γ=0) runs. SFT + SpanFDD baselines add another ~30 GPU-hours.

## Rollback Plan

Each run is isolated by `--save` directory. Failed runs can be re-launched without affecting others. If a phase-wide bug is found mid-batch, kill all jobs (SIGTERM), fix Phase 3 code, restart.

## Next Steps / Dependencies

- **Blocked by:** Phase 3 (trainer code), Phase 2 (projector checkpoints).
- **Unblocks:** Phase 5 (eval consumes these checkpoints).

## Unresolved Questions

- None for this phase. Compute may exceed budget; if so, drop seeds 1337 / 2024 from non-primary configs and rely on 1-seed estimates with explicit caveat.

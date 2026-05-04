# Phase 02 — Stage 1 — Projector Pretraining (P_T→A, P_A→T)

## Context Links

- Brainstorm spec §3.1
- Paper: On et al., 2026 Section 3.1 (projector pretraining objective)
- Reference trainer scaffold: `MTA/distillm-master/finetune.py` (skeleton), `span_finetune.py` (deepspeed init pattern)
- Tokenizer + dataset: `data_utils/lm_datasets.py::LMTrainDataset`

## Overview

- **Priority:** P1
- **Status:** in-progress (code written 260503; runs pending)
- **Teacher:** `VoCuc/Qwen1.5_1.8B_SFT_Dolly` — single teacher for both Setup A + B. d_T=2048, d_A=64.
- **Description:** Train two linear maps `P_T→A: d_T → d_A` and `P_A→T: d_A → d_T` such that `softmax(W_T · P_A→T(P_T→A(h^T)))` predicts next-token correctly. Teacher hidden states + W_T (LM head) are frozen.

## Key Insights from Scouting

- Teacher `VoCuc/Qwen1.5_1.8B_SFT_Dolly` loaded bfloat16, frozen. `config.hidden_size=2048`.
- Phase 2 has NO student model — `--model-path` points to teacher so `get_tokenizer` loads Qwen tokenizer.
- Data: `processed_data/dolly/full/qwen/` (Qwen-tokenised binaries, already on disk).
- `LMTrainDataset` with `model_type=qwen` uses `4294967295` as prompt/response separator.
- Only P_T→A + P_A→T params (~262K) are trainable; teacher LM head used for CE but frozen.
- `deepspeed.initialize(model=projector, ...)` wraps the projector directly (no student model needed).

## Requirements

**Functional:**
1. Projector module `class ProjectorTA(nn.Module)`: 2 layers `Linear(d_T, d_A)` + `Linear(d_A, d_T)`, no bias by default (matches paper); save/load via `state_dict`.
2. Pretraining loop: forward teacher (frozen, no_grad) → get `h^T` (last layer) → reconstruct → CE against next-token label.
3. Hyperparameters per brainstorm: 10 epochs, lr=1e-3, AdamW, cosine, weight_decay=1e-4, warmup_ratio=0.1.
4. Loss masking: only response tokens (i.e. positions where `label != -100`).
5. Checkpoint save: `{save_path}/projector_TA.pt` and `projector_AT.pt` after final epoch + best-val.

**Non-functional:**
- Single 16-40GB GPU sufficient (teacher in fp16, projectors fp32, batch ≤ 16).
- < 200 LOC for `span_residual_pretrain.py`; helpers in `span_residual_utils.py`.

## Architecture

Data flow per step:
```
input_ids  → Teacher (frozen, fp16)
            → hidden_states[-1]  (B, L, d_T)
            → P_T→A             → (B, L, d_A)
            → P_A→T             → (B, L, d_T)
            → W_T (frozen)       → logits (B, L, V_T)
            → CE(logits, no_model_batch.label)
            → backward only on P_T→A, P_A→T params
```

Param count: GPT2 d_T=1600, d_A=64 → P_T→A 102K + P_A→T 102K ≈ 204K. Qwen d_T=2048, d_A=64 → 262K. Both trivial.

## Related Code Files (as of 260503)

**Modified:**
- `MTA/distillm-master/arguments.py` — added `--d-bottleneck`, `--projector-save-path`, `--projector-load-path`, `--projector-pretrain-epochs`, `--projector-lr`, `--lambda-res`, `--gamma-span`, `--teacher-data-dir`. Additive only.

**Created:**
- `MTA/distillm-master/span_residual_utils.py` — `ProjectorTA`, `ProjectorSA`, `cross_model_attention`, `compute_beta_seq`, `compute_residual_mask`, `compute_residual_correction`, `load_projectors`. ~140 LOC.
- `MTA/distillm-master/span_residual_pretrain.py` — Stage-1 trainer (~190 LOC). Loads Qwen teacher, trains ProjectorTA only, saves projector_best.pt.
- `MTA/distillm-master/span_residual_finetune.py` — copy of `span_fdd_finetune.py` (Stage 2 edits deferred to Phase 3).
- `MTA/distillm-master/scripts/gpt2/spanresidual/pretrain-qwen1.8B-projectors.sh` — runs `span_residual_pretrain.py` with Qwen1.5-1.8B teacher.
- `MTA/distillm-master/scripts/gpt2/spanresidual/train-setup-A-0.1B-qwen1.8B.sh` — Stage-2 placeholder for Qwen→GPT2-120M.
- `MTA/distillm-master/scripts/gpt2/spanresidual/train-setup-B-0.35B-qwen1.8B.sh` — Stage-2 placeholder for Qwen→GPT2-medium.

**Delete:** none.

## Implementation Steps

1. **Add args to `arguments.py`** in `add_distillm_args` group:
   ```
   --d-bottleneck (int, default=64)
   --projector-save-path (str, default=None)
   --projector-load-path (str, default=None)   # consumed by Phase 3
   --projector-pretrain-epochs (int, default=10)
   --projector-lr (float, default=1e-3)
   ```
2. **Implement `span_residual_utils.py::ProjectorTA`:**
   ```
   class ProjectorTA(nn.Module):
       def __init__(self, d_T, d_A):
           super().__init__()
           self.P_TA = nn.Linear(d_T, d_A, bias=False)
           self.P_AT = nn.Linear(d_A, d_T, bias=False)
       def forward(self, h):
           z = self.P_TA(h)
           return z, self.P_AT(z)
   ```
3. **Write `span_residual_pretrain.py`:**
   - Parse args; init distributed; load tokenizer + dataset (`prepare_dataset`).
   - Load teacher (frozen) via `get_teacher_model`. Set `requires_grad=False` on all params.
   - Build `ProjectorTA(d_T, args.d_bottleneck).to(device)` (fp32).
   - Optimizer: AdamW on projector params only, lr=`args.projector_lr`, wd=1e-4.
   - Scheduler: CosineAnnealingLR over `total_iters`.
   - Use DeepSpeed wrapper: `deepspeed.initialize(model=projector, ...)` — projector is the trainable model.
   - Training loop:
     - `with torch.no_grad(): teacher_out = teacher(**model_batch, output_hidden_states=True)`
     - `h_T = teacher_out.hidden_states[-1]`  (B, L, d_T)
     - `_, h_recon = projector(h_T)`
     - `logits = teacher.lm_head(h_recon)`
     - `loss = CE(logits.view(-1, V), label.view(-1))`  (label has -100 masking)
     - `model.backward(loss); model.step()`
   - Validation each epoch on dev split → log val loss.
   - Save `state_dict()` to `{save_path}/projector_epoch{N}.pt` + `projector_best.pt`.
4. **Write `pretrain-0.1B-1.5B.sh`** mirroring `spanfdd/train_0.1B_1.5B.sh`:
   - Replace entry point with `span_residual_pretrain.py`.
   - Set `--lr 1e-3 --weight-decay 1e-4 --epochs 10 --warmup-ratio 0.1 --lr-decay-style cosine`.
   - `--save_path = ${BASE_PATH}/results/gpt2/projectors/spanresidual_1.5B/`.
   - Drop irrelevant args (`--type`, `--student-gen`, layer mappings).
5. **Write `pretrain-0.5B-1.8B.sh`** analogous for Qwen.
6. **Run Stage 1 — GPT2** (~3-5h on 1× A100/A40):
   ```
   bash scripts/gpt2/spanresidual/pretrain-0.1B-1.5B.sh
   ```
7. **Run Stage 1 — Qwen** (~3-5h):
   ```
   bash scripts/qwen1.5/spanresidual/pretrain-0.5B-1.8B.sh
   ```
8. **Sanity check:** Final reconstruction CE < 2.0 (cf. teacher SFT CE ≈ 1.7) → success. If higher, log epoch-by-epoch curve and assess if more epochs needed.

## Todo Checklist

- [x] Add new args to `arguments.py` (d-bottleneck, projector-*, lambda-res, gamma-span, teacher-data-dir)
- [x] Implement `span_residual_utils.py` (ProjectorTA, ProjectorSA, cross_model_attention, helpers)
- [x] Implement `span_residual_pretrain.py` (Stage 1 trainer)
- [x] Write `pretrain-qwen1.8B-projectors.sh`
- [x] Write `train-setup-A-0.1B-qwen1.8B.sh` and `train-setup-B-0.35B-qwen1.8B.sh` (Stage-2 placeholders)
- [ ] Run Stage 1 pretrain → verify final val CE < 2.5
- [ ] Confirm `projector_best.pt` exists and loads cleanly

## Success Criteria

- Both `projector_best.pt` checkpoints exist and load via `torch.load`.
- Reconstruction CE on dev: GPT2 < 2.5; Qwen < 2.5 (measured against teacher's own SFT CE on Dolly dev).
- Param count printed at startup: ≤ 300K (sanity that we're not training teacher).

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Teacher hidden state shape mismatch with W_T (e.g. tied embeddings) | M | M | Handle via `model_type` branch; for GPT2 use `lm_head` weight tied to `wte`; for Qwen `lm_head` is separate |
| Pretrain CE plateaus high (>3.0) | L | H | Increase d_bottleneck to 128 (paper Fig 2a shows minor variance); or extend to 15 epochs |
| 10 epochs × 11k samples too slow on single GPU | M | M | Batch 16, grad-accum 1 → ~5 min/epoch on A100; Total ~1h. Fall-back: reduce to 5 epochs (sanity loss curve at 3, 5, 10) |
| Out-of-memory caching teacher hidden | L | M | Teacher in fp16, no_grad, output_hidden_states=False except last → only last layer needed; reduce batch if needed |
| DeepSpeed wrapper breaks for projectors-only training | L | M | Fall back to plain `model = projector.to(device); optimizer.step()` (no DS) — adequate for 200K params |

## Rollback Plan

Stage-1 outputs are isolated under `results/{model}/projectors/`. Wipe folder and re-run; no impact on other phases. If projector quality is poor, ablate γ=0 path in Phase 5 still works (no projector needed for pure Residual baseline reproduction since we re-run their setup with our impl).

## Next Steps / Dependencies

- **Blocked by:** Phase 1 (needs `processed_data/dolly/full/{gpt2,qwen}/`).
- **Unblocks:** Phase 3 (loads `projector_best.pt` via `--projector-load-path`).

## Unresolved Questions

- **Q1 partial (P_A→S init):** This phase only outputs P_T→A and P_A→T. Phase 3 will introduce P_A→S separately; default learnable from scratch, ablation warm-starts from P_A→T (only valid when d_S = d_T which never holds in our setups, so ablation falls back to "init via Xavier of same shape as P_A→T transposed"). Document in Phase 3.
- **Q2 (β level):** Resolved — sequence-level β computed in Phase 3 via `compute_beta_seq` defined here.
- **Q4 (Pretrain data):** Resolved — Dolly only.

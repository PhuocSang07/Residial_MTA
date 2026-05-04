# Phase 03 — Stage 2 — SpanResidual Trainer (Cross-Tokenizer)

## Context Links

- Brainstorm spec §3.2
- Paper §3.4: cross-model attention Eq. 9-10 (On et al. ICLR 2026)
- Base file to modify: `MTA/distillm-master/span_residual_finetune.py` (copy of `span_fdd_finetune.py`)
- Utilities: `MTA/distillm-master/span_residual_utils.py` (ProjectorTA, ProjectorSA, cross_model_attention, etc.)
- Phase 2 output: `results/qwen/projectors/spanresidual_qwen1.8B/projector_best.pt`

## Overview

- **Priority:** P1
- **Status:** pending
- **Scope:** Cross-tokenizer only — Qwen1.5-1.8B teacher (vocab=152064) → GPT2-120M or GPT2-medium student (vocab=50257). n_T ≠ n_S per batch.
- **Description:** Implement `span_residual_finetune.py` combining: (a) student CE L_SFT; (b) residual-corrected CE L_res using cross-model attention to align teacher hiddens to student positions; (c) MTA span loss L_SpanMTA on student sequence only.

## Key Insights from Scouting

- **Dual dataloader:** Teacher (Qwen) and student (GPT2) tokenise the same raw text differently → two separate `LMTrainDataset` instances from different `data_dir`s. Must iterate both in lockstep per batch.
- **Cross-model attention (paper Eq. 9-10):** `cross_model_attention(h_S_A, h_T_A)` in `span_residual_utils.py` aligns (B, n_T, d_A) teacher hidden to (B, n_S, d_A) student positions.
- **Residual chain (cross-tokenizer):**
  1. `h_T_A = projector_TA.encode(h_T)` → (B, n_T, d_A) — frozen P_T→A
  2. `h_S_A = projector_SA(h_S_last)` → (B, n_S, d_A) — learnable P_S→A
  3. `h_T_aligned = cross_model_attention(h_S_A, h_T_A)` → (B, n_S, d_A)
  4. `proj_to_S = projector_AS(h_T_aligned)` → (B, n_S, d_S) — learnable P_A→S
  5. `beta` = sequence-level norm ratio (see below)
  6. `resp_mask` = student response mask `(student_label != -100)`
  7. `h_S_res = h_S_last - beta * proj_to_S * resp_mask.unsqueeze(-1)`
  8. `L_res = CE(student.lm_head(h_S_res), student_label)`
- **acc_mask simplification:** Cross-tokenizer: teacher vocab ≠ student vocab → can't compare teacher argmax to student label directly. Use response mask only (`resp_mask`) as acc_mask. Annotate as TODO for future: align via char-level text match.
- **Span loss (student-side only):** `compute_overall_span_loss` uses student sequence hidden states and student tokenizer offsets. Teacher seq is NOT passed to span loss.
- **vocab resize line must be removed:** `span_fdd_finetune.py:945` has `teacher_model.resize_token_embeddings(model.config.vocab_size)` — this MUST NOT appear in `span_residual_finetune.py` (cross-tokenizer: vocabularies must remain separate).
- **`model_type` for dataset:** Student dataloader uses `model_type=gpt2`; teacher dataloader uses `model_type=qwen`. Different `LMTrainDataset` instances.

## Requirements

**Functional:**
1. Dual dataloader: student `LMTrainDataset(args, student_tok, args.data_dir, ...)` + teacher `LMTrainDataset(args_teacher, teacher_tok, args.teacher_data_dir, ...)`. Iterate lockstep; drop teacher batch if shorter.
2. Load P_T→A frozen from `args.projector_load_path` via `load_projectors`.
3. Build P_S→A (`ProjectorSA(d_S, d_A)`) and P_A→S (`nn.Linear(d_A, d_S)`), both learnable.
4. Cross-model attention for (B, n_S, d_A) ← aligned from (B, n_T, d_A).
5. β: `beta = sqrt(d_S/d_A) · mean(||h_S||/||proj_to_S||) over response tokens`, detached.
6. `L_res = CE(student.lm_head(h_S - beta·proj_to_S·resp_mask), student_label)`.
7. `L_SFT = CE(student_logits, student_label)`.
8. `L_span = compute_overall_span_loss(...)` on student hidden states + student tokenizer offsets.
9. `L = (1-λ)·L_SFT + λ·L_res + γ·L_span`. CLI: `--lambda-res 0.5 --gamma-span 1.0`.
10. Log L_SFT, L_res, L_span, β each step for first 100 steps; then every log_interval.
11. At startup: assert teacher and student tokenizers differ (sanity check cross-tokenizer is active).

**Non-functional:**
- `span_residual_finetune.py` < 350 LOC (modularise helpers in `span_residual_utils.py`).
- Do NOT edit `span_fdd_finetune.py` or `span_utils.py` (backward compat).
- Projector P_T→A attached as `model.module.projector_TA` after `deepspeed.initialize` (buffer, not param).

## Architecture

```
Per training step (cross-tokenizer):

student_batch (GPT2-tokenised)     teacher_batch (Qwen-tokenised)
        |                                   |
   student model ─────────────────────── teacher model (frozen)
   s_logits (B, n_S, V_S)           t_hidden_states[-1] (B, n_T, d_T)
   s_hidden[-1] (B, n_S, d_S)
        |                                   |
        |                          projector_TA.encode(h_T)  ─ frozen P_T→A
        |                               h_T_A (B, n_T, d_A)
        |                                   |
   projector_SA(h_S_last)         cross_model_attention(h_S_A, h_T_A)
   h_S_A (B, n_S, d_A)               h_T_aligned (B, n_S, d_A)
        \                             /
         projector_AS(h_T_aligned)
         proj_to_S (B, n_S, d_S)
              |
         beta * proj_to_S * resp_mask
              |
         h_S_res = h_S_last - correction
              |
         L_res = CE(lm_head(h_S_res), student_label)
         L_SFT = CE(s_logits, student_label)
         L_span = compute_overall_span_loss(student-side)
              |
         L = (1-λ)·L_SFT + λ·L_res + γ·L_span
```

## Related Code Files

**Modify:**
- `MTA/distillm-master/span_residual_finetune.py` — the Phase 3 trainer. Currently a verbatim copy of `span_fdd_finetune.py`. This is the primary file to edit.

**Update (additive):**
- `MTA/distillm-master/span_residual_utils.py` — may need additional helpers (`make_teacher_args`, `load_teacher_tokenizer`).

**Do NOT modify:**
- `span_fdd_finetune.py`, `span_utils.py`, `arguments.py` (already extended), `data_utils/`.

## Implementation Steps

1. **Dual dataloader setup in `prepare_dataset`:**
   ```python
   def prepare_dataset(args, student_tokenizer, teacher_tokenizer):
       rng = random.Random(args.seed)
       data = {
           "train": LMTrainDataset(args, student_tokenizer, args.data_dir, "train", ...),
           "dev":   LMTrainDataset(args, student_tokenizer, args.data_dir, "valid", ...),
       }
       if args.teacher_data_dir:
           args_t = copy.copy(args); args_t.model_type = args.teacher_model_type or "qwen"
           data["teacher_train"] = LMTrainDataset(args_t, teacher_tokenizer, args.teacher_data_dir, "train", ...)
       return data
   ```
   Iterate: `zip(train_dataloader, teacher_train_dataloader)` — shortest truncates.

2. **Load teacher tokenizer separately:**
   ```python
   student_tokenizer = get_tokenizer(args)                        # GPT2
   teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)  # Qwen
   if teacher_tokenizer.pad_token_id is None:
       teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
   ```

3. **Model setup (remove vocab-resize):**
   ```python
   model = get_model(args, device)            # student (GPT2)
   teacher = get_teacher_model(args, device)  # Qwen — do NOT resize embeddings
   d_S = model.config.n_embd                  # GPT2: 768 or 1024
   d_T = teacher.config.hidden_size           # Qwen: 2048
   d_A = args.d_bottleneck                    # 64
   ```

4. **Attach projectors:**
   ```python
   # Span projectors (same as span_fdd_finetune.py lines 957-965)
   projector_list = nn.ModuleList([nn.Linear(d_S, d_T) for _ in teacher_layer_mapping])
   model.projectors = projector_list
   # Residual projectors (learnable)
   model.projector_SA = ProjectorSA(d_S, d_A)
   model.projector_AS = nn.Linear(d_A, d_S, bias=False)
   nn.init.xavier_uniform_(model.projector_AS.weight)
   ```
   After `deepspeed.initialize`, attach frozen P_T→A:
   ```python
   projector_TA = load_projectors(args.projector_load_path, d_T, d_A, device)
   model.module.projector_TA = projector_TA  # buffer only, no gradient
   ```

5. **Training loop changes in `finetune()`:**
   - Accept `teacher_dataloader` as extra arg; zip with `train_dataloader`.
   - Student forward: `outputs = model(**s_batch, output_hidden_states=True, use_cache=False)`.
   - Teacher forward (no_grad): `t_out = teacher(**t_batch, output_hidden_states=True, use_cache=False)`.
   - Compute cross-tokenizer residual:
     ```python
     h_S = outputs.hidden_states[-1]           # (B, n_S, d_S)
     h_T = t_out.hidden_states[-1]             # (B, n_T, d_T)
     with torch.no_grad():
         h_T_A = model.module.projector_TA.encode(h_T.float())  # (B, n_T, d_A) frozen
     h_S_A = model.projector_SA(h_S.float())   # (B, n_S, d_A)
     h_T_aligned = cross_model_attention(h_S_A.detach(), h_T_A)  # (B, n_S, d_A)
     proj_to_S = model.projector_AS(h_T_aligned)  # (B, n_S, d_S)
     resp_mask = (s_no_model_batch["label"] != -100)
     beta = compute_beta_seq(h_S.detach(), proj_to_S.detach(), resp_mask, d_S, d_A)
     L_res = compute_residual_loss(model.lm_head, h_S, proj_to_S, beta, resp_mask, s_no_model_batch["label"])
     ```
   - Keep span loss from existing pattern but on student sequence only.
   - Drop: teacher vocab resize, `replay_buffer`, `student_gen`, `SampleGenerator`.

6. **50-iter smoke test** (both setups):
   ```bash
   # Setup A
   bash scripts/gpt2/spanresidual/train-setup-A-0.1B-qwen1.8B.sh \
       --total-iters 50 --log-interval 5 --save-interval -1 --eval-interval -1
   ```
   Verify: all 3 losses finite; β ∈ (0.1, 10); no OOM.

7. **γ=0 sanity run** (50 iters): confirms L_span = 0 and no spaCy calls when `--gamma-span 0`.

## Todo Checklist

- [ ] Rewrite `prepare_dataset` in `span_residual_finetune.py` for dual tokenizer + dual dataloader
- [ ] Add `load_teacher_tokenizer` helper (or inline in main)
- [ ] Remove `teacher_model.resize_token_embeddings(...)` line
- [ ] Attach P_S→A + P_A→S to model; include in optimizer with lr=5e-4
- [ ] Load frozen P_T→A via `load_projectors`, attach as `model.module.projector_TA`
- [ ] Implement cross-tokenizer residual step in `finetune()` using `cross_model_attention`
- [ ] Compute `L_SFT`, `L_res`, `L_span` and compose with `--lambda-res`, `--gamma-span`
- [ ] Per-step logging for first 100 steps (L_SFT, L_res, L_span, β)
- [ ] Drop replay_buffer / student_gen / adaptive from `finetune()`
- [ ] 50-iter smoke test (Setup A) → no NaN
- [ ] 50-iter smoke test (Setup B)
- [ ] γ=0 path test → L_span == 0

## Success Criteria

- Smoke test (50 iters): all loss components finite, β ∈ (0.1, 10).
- `span_residual_finetune.py` LOC < 350.
- Setting `--gamma-span 0` skips spaCy entirely (fast path).
- Teacher and student tokenizer sizes differ (assert at startup passes).
- Both Setup A and B run without OOM on RTX PRO 6000 Blackwell 102GB.

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Dual dataloader batch mismatch (different seq len per step) | M | M | LMTrainDataset pads to max_length; both always (B, 256); lockstep zip safe |
| P_T→A frozen but still tracked by DeepSpeed | M | M | Attach AFTER `deepspeed.initialize` via `model.module.projector_TA`; use `torch.no_grad()` for all P_T→A calls |
| β explodes (large d_T/d_A ratio) | M | H | Clamp β to [0.05, 10]; log warning if out of (0.2, 5) |
| Cross-model attention softmax over n_T positions → uniform if n_T=256 | L | M | Standard for Eq. 9-10; paper validated it |
| acc_mask all-False (teacher not wrong enough) | L | L | Using resp_mask only → this is the cross-tokenizer fallback |
| Memory: two sets of hidden states (n_T + n_S) simultaneously | M | M | Both capped at max_length=256; teacher in fp16; projectors fp32 — total < 4GB extra VRAM |

## Rollback Plan

`span_residual_finetune.py` is an isolated new file. Wipe and revert to copy of `span_fdd_finetune.py`. No other files affected.

## Next Steps / Dependencies

- **Blocked by:** Phase 2 (needs `projector_best.pt`).
- **Unblocks:** Phase 4 (full training runs).

## Unresolved Questions

- **Q-cross-acc-mask:** Using `resp_mask` as proxy for acc_mask. More precise: align teacher acc_mask via cross-attention matrix A (token-level interpolation). Deferred to Phase 5 ablation.
- **Q-beta-detach:** β is detached (`beta.detach()`) per paper — residual correction is a fixed offset, not a learnable scaling. Confirm in Phase 4 loss curves that β stays stable.

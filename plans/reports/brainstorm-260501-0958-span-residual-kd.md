# Brainstorm: SpanResidual KD — MTA Span ⊕ Residual Learning

**Date:** 2026-05-01
**Paper baseline:** On et al., 2026 — *Knowledge Distillation for Large Language Models through Residual Learning* (ICLR 2026)
**MTA baseline:** Span mechanism (currently plugged vào FDD/DistiLLM/CSD)
**Goal:** Plug MTA span vào Residual Learning baseline → vượt Residual baseline.

---

## 1. Problem Statement

Residual Learning (paper) attack 2 vấn đề: **teacher hacking** + **teacher bias** bằng cách học residual `h^S − β·P(h^T)` ở token-level chỉ tại positions teacher dự đoán sai. Tuy nhiên paper:

- Chỉ supervise ở **last hidden layer**, không tận dụng intermediate layers
- Residual gate là **token-level binary mask** (`1[teacher_wrong]`) — coarse
- Không tận dụng **structural prior** (noun phrases, verb phrases) là nơi reasoning điển hình tập trung

**Hypothesis:** MTA span (multi-layer span pooling + intra-batch span similarity + projector cosine ở các layer trung gian) cung cấp supervision **bổ sung trực giao** với residual learning. Cả hai cùng chống teacher imperfection nhưng ở **các phần khác nhau của network** (residual ở output/last-layer, span ở intermediate representation).

---

## 2. Evaluated Approaches

### Option A — Span-Auxiliary Residual ✅ CHỌN

`L = (1−λ)·L_SFT + λ·L_res + γ·L_SpanMTA`

- L_res, L_SFT: theo paper Eq. 6 + Eq. 11
- L_SpanMTA: existing `compute_overall_span_loss` từ `MTA/distillm-master/span_utils.py`
- 2 component **độc lập**: residual ở token-level (last hidden); span ở multi-layer (intermediate)
- λ=0.5 (paper), γ ablate {0, 0.5, 1.0, 2.0}

**Pros:** isolate được contribution của span; baseline so sánh sạch (γ=0 = paper); reuse code.
**Cons:** ít "novel" hơn B/C; 2 mechanism không tương tác trực tiếp.

### Option B — Span-Gated Residual (KHÔNG CHỌN)

Thay accuracy mask `1[teacher_wrong]` bằng span-aware gate (span CE trung bình > threshold) + pretrain projector cộng span-reconstruction objective.

**Cons:** phức tạp hơn 2-3x, deviate xa paper, khó attribute nguyên nhân improvement.

### Option C — Span-Level Residual (KHÔNG CHỌN)

Residual đặt trực tiếp trên span-pooled hidden, dùng span hidden cho next-token prediction.

**Cons:** span hidden không trực tiếp predict next-token; cần loss surrogate; deviate quá xa paper.

---

## 3. Recommended Solution: SpanResidual KD (Option A)

### 3.1 Stage 1 — Projector Pretraining (paper-faithful)

**Objective:** học P_T→A (compress teacher hidden → d_A=64) và P_A→T (decompress) sao cho `W_T · P_A→T(P_T→A(h^T))` predict next-token đúng.

```
L_pretrain = CE( softmax(W_T · P_A→T(P_T→A(h^T))) , next_token )
```

- 10 epochs, lr=1e-3, AdamW, Cosine decay, weight decay=1e-4
- Dataset: Dolly train split (~11k samples)
- Save P_T→A, P_A→T checkpoints
- Stage 2: P_T→A frozen, P_A→T optionally frozen

### 3.2 Stage 2 — SpanResidual Distillation

**A. Compute residual hidden (paper Eq. 4-5):**

```
β = sqrt(d_S/d_A) · mean_{i=1..n}( ||h^S_i|| / ||P_A→S(P_T→A(h^T_i))|| )
mask_i = 1[ argmax(W_T · h^T_i) ≠ y_i ]
˜h^S_i = h^S_i − β · P_A→S(P_T→A(h^T_i)) · mask_i
```

- P_A→S: new learnable linear (d_A → d_S) — same-tokenizer thì có thể reuse weights P_A→T nếu d_S=d_T, nhưng ta keep generic (cross-architecture sẽ cần)
- Mask only on **response tokens** (loại prompt tokens, padding tokens)

**B. Residual loss:**

```
˜z_i = W_S · ˜h^S_i
L_res = CE(˜z, y)
```

**C. SFT loss:**

```
L_SFT = CE(W_S · h^S, y)   # vanilla cross-entropy
```

**D. MTA Span loss (multi-layer, existing):**

- spaCy → noun_chunks + verb phrases → spans_offsets
- For each (s_layer, t_layer) trong layer_mapping:
  - Span-pooled hidden via attention-weighted token weights
  - Intra-batch span-similarity matching loss + projector cosine loss (xem `compute_hidden_span_loss`)
- L_SpanMTA = sum / num_layers

**E. Total:**

```
L = (1−λ)·L_SFT + λ·L_res + γ·L_SpanMTA
```

- Default: λ=0.5, γ=1.0
- Ablation: γ ∈ {0, 0.5, 1.0, 2.0}; γ=0 → paper baseline (Residual)

### 3.3 Models (1-2 GPU, 16-40GB)

| Setup | Teacher | Student | Tokenizer | LoRA | GPU mem |
|-------|---------|---------|-----------|------|---------|
| Run 1 | GPT2-1.5B (SFT) | GPT2-0.1B | same | no | ~16GB |
| Run 2 | Qwen1.5-1.8B (SFT) | Qwen1.5-0.5B | same | no | ~30GB |

Both đã có script tham khảo trong `MTA/distillm-master/scripts/{gpt2,qwen1.5}/spanfdd/`.

### 3.4 Evaluation

- Datasets: Dolly (500), SelfInst, VicunaEval, S-NI, UnNI — paper protocol
- Metric: Rouge-L %
- 3 random seeds → mean ± std
- Baselines:
  1. **SFT** only
  2. **Residual** (paper, no MTA span — γ=0)
  3. **SpanFDD** (existing MTA, no residual)
  4. **SpanResidual** (ours)

---

## 4. Implementation Plan (cao độ)

### Files to create (`MTA/distillm-master/`)

```
span_residual_pretrain.py        # Stage 1: pretrain P_T→A, P_A→T
span_residual_finetune.py        # Stage 2: distillation
span_residual_utils.py           # residual mask, β computation, P_A→S
                                 # (extend span_utils.py if cleaner)
scripts/gpt2/spanresidual/
  pretrain_0.1B_1.5B.sh
  train_0.1B_1.5B.sh
scripts/qwen1.5/spanresidual/
  pretrain_0.5B_1.8B.sh
  train_0.5B_1.8B.sh
```

### Reused from existing MTA codebase

- `span_utils.py` → `compute_overall_span_loss`, `get_spans_offsets`, spaCy pipeline
- `span_finetune.py` → CLI args, dataset loading, trainer loop pattern
- `MTA/src/evaluator.py` → eval logic
- Existing `scripts/eval_*.sh` patterns

### Hyperparameters (paper Table 4)

| Param | Value |
|-------|-------|
| d_A | 64 |
| λ (residual weight) | 0.5 |
| γ (span weight) | 1.0 (sweep 0/0.5/1/2) |
| Pretrain epochs | 10 |
| Pretrain LR | 1e-3 |
| Distill LR (student) | 5e-6 (Qwen) / 1e-3 (GPT2) |
| Batch size | 128 (gradient accumulate) |
| Max seq len | 512 |
| Distill epochs | 3-10 |
| Optimizer | AdamW |
| Scheduler | Cosine |
| Warmup ratio | 0.1 |
| Weight decay | 1e-4 |

---

## 5. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Projector pretrain slow (10 epochs × 11k samples) | Cache pretrained weights; ablate fewer epochs (3, 5) |
| spaCy span extraction CPU bottleneck | Pre-extract spans to disk (one-pass over Dolly) |
| Cross-arch P_A→S unstable nếu d_A=64 quá nhỏ cho student dim | Paper Fig. 2(a) confirm d_A=64 best; nếu fail thử d_A=128 |
| λ, γ tuning explodes search space | Fix λ=0.5 (paper), only sweep γ |
| Teacher-wrong mask quá sparse (teacher pretty good) | Paper "w/o accuracy mask" only -0.42 → có thể fallback nếu cần |
| Same-tokenizer setup: P_A→S vs P_A→T trùng | Same-tokenizer + same-arch: reuse P_A→T weights init cho P_A→S (warm start) |
| MTA span weights và residual β scale conflict | Normalize loss components về cùng order trước khi cộng (log magnitude check trong first 100 steps) |

---

## 6. Success Metrics

- **Primary:** SpanResidual avg Rouge-L > Residual baseline (γ=0) on cả 2 setup
- **Secondary:** SpanResidual > SpanFDD (chứng minh residual + span tốt hơn FDD + span)
- **Tertiary:** Reproduce paper's Residual baseline trong ±0.5 Rouge-L (validate implementation)
- **Ablation:** γ sweep cho thấy non-zero γ optimal

---

## 7. Validation Strategy

1. **Sanity:** Sau Stage 1, projector reconstruction CE < 2.0 (teacher SFT level)
2. **Sanity:** Stage 2 first 100 steps — log L_res, L_SFT, L_SpanMTA magnitudes; nếu chênh >10x → adjust γ
3. **Reproduce:** Run γ=0 → so với paper Table 1/2 row "Ours" (Mistral) hoặc đo so với SpanFDD MTA
4. **Ablation:** γ=0 vs γ=1 trên 1 seed Dolly only → xác nhận trend trước khi full eval

---

## 8. Next Steps

1. Confirm design (xong)
2. Tạo implementation plan chi tiết qua `/ck:plan`
3. Stage 1 implement + pretrain projectors trên cả 2 setup
4. Stage 2 implement + train SpanResidual
5. Eval suite (Rouge-L 5 benchmarks × 3 seeds × 2 setup × 4 baselines)
6. Ablation γ sweep
7. Write up results

---

## 9. Unresolved Questions

1. **P_A→S init:** same-tokenizer + same-arch (e.g. GPT2→GPT2) thì P_A→S = P_A→T về dim. Reuse weights hay learnable từ scratch? (đề xuất: warm-start với P_A→T, fine-tune)
2. **β at sequence vs token level:** paper "empirically observe sequence-level β stable" — confirm dùng sequence-level mean
3. **Span layer mapping:** existing MTA dùng `student_layer_mapping`/`teacher_layer_mapping`. Paper Residual chỉ dùng last hidden. Nên align span ở layer nào? Đề xuất: dùng `split_layer_mapping` mặc định MTA, ablate sau
4. **Pretraining data:** dùng Dolly hay UltraInteract (paper)? Đề xuất Dolly (đã có, fair với MTA scripts)
5. **Mask granularity:** chỉ áp residual ở response tokens (paper) hay cả prompt+response? Paper không nói rõ — đề xuất chỉ response để giảm noise
6. **Eval datasets:** SelfInst/VicunaEval/S-NI/UnNI có sẵn trong MTA chưa? Cần verify trước Stage 2

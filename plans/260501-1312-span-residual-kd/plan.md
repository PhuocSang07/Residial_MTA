---
title: "SpanResidual KD — MTA Span ⊕ Residual Learning"
description: "Two-stage white-box KD combining residual learning (On et al., ICLR 2026) with MTA span supervision."
status: pending
priority: P1
effort: ~5-7d compute + ~2d code
branch: main
tags: [knowledge-distillation, residual-learning, mta, span, llm]
created: 2026-05-01
---

# SpanResidual KD — Plan Overview

**Work context:** `/teamspace/studios/this_studio/mlj-exp/`
**Code root:** `mlj-exp/MTA/distillm-master/` (all relative paths in phase files resolve under this root)
**Reports:** `mlj-exp/plans/reports/`
**Origin:** plan authored 2026-05-01 in `mta/`; copied to `mlj-exp/` on 2026-05-03 to execute against this codebase variant (which has additional `*_entropy.sh` / `span_fdd_finetune_entropy.py` files vs `mta/`).

**Goal:** Plug MTA span loss into Residual Learning (paper) → beat Residual baseline + SpanFDD baseline on Dolly/SelfInst/VicunaEval/S-NI/UnNI.

**Loss:** `L = (1−λ)·L_SFT + λ·L_res + γ·L_SpanMTA`, λ=0.5, γ=1.0 default.

**Authoritative spec:** `plans/reports/brainstorm-260501-0958-span-residual-kd.md`.
**Reference paper:** On et al., 2026 — *KD for LLMs through Residual Learning* (ICLR 2026).

## Phases

| # | Phase | Status | Owner | Effort | Blocks |
|---|-------|--------|-------|--------|--------|
| 1 | [Setup, data preprocessing, eval benchmarks](./phase-01-setup-and-data.md) | completed (260503) | — | done | 2,3,4,5 |
| 2 | [Stage 1 — Projector pretraining (P_T→A, P_A→T)](./phase-02-stage1-projector-pretrain.md) | in-progress (code done 260503; runs pending) | — | ~6h compute | 3,4 |
| 3 | [Stage 2 — SpanResidual trainer (`span_residual_finetune.py`)](./phase-03-stage2-spanresidual-trainer.md) | pending (cross-tokenizer redesign documented 260503) | — | 1d | 4,5 |
| 4 | [Training scripts + actual runs](./phase-04-training-scripts-and-runs.md) | pending | — | 2-3d compute | 5 |
| 5 | [Evaluation + ablation (Rouge-L × 5 benchmarks × 4 baselines × 2 setups × 3 seeds)](./phase-05-evaluation-and-ablation.md) | pending | — | 1-2d | 6 |
| 6 | [Results writeup + tables](./phase-06-results-writeup.md) (optional) | pending | — | 0.5d | — |

## Critical Path

`Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → (Phase 6)`

Phase 2 produces P_T→A / P_A→T checkpoints used in Phase 3 (residual mask hidden). Phase 3 produces trainer code used in Phase 4 (runs). Phase 4 outputs distilled checkpoints consumed by Phase 5.

## File Ownership Map (no overlap across phases)

| Phase | Owns (modify/create) |
|-------|---------------------|
| 1 | `MTA/distillm-master/processed_data/`, `MTA/distillm-master/data/`, span cache scripts under `tools/` |
| 2 | `MTA/distillm-master/span_residual_pretrain.py`, `span_residual_utils.py` (P_T→A/P_A→T defs), `scripts/{gpt2,qwen1.5}/spanresidual/pretrain_*.sh` |
| 3 | `MTA/distillm-master/span_residual_finetune.py`, `span_residual_utils.py` (residual + β + L_res), `arguments.py` (new args only — additive, no rename) |
| 4 | `scripts/{gpt2,qwen1.5}/spanresidual/train_*.sh`, run logs |
| 5 | `scripts/eval/*.sh`, `tools/eval_aggregate.py`, eval logs |
| 6 | `plans/260501-1312-span-residual-kd/results.md` |

## Key Dependencies

- **Phase 1 must verify** Dolly processed (`processed_data/dolly/full/{gpt2,qwen}/`) AND eval benchmarks (SelfInst, VicunaEval, S-NI, UnNI) present before any training. Acquisition steps documented if missing.
- **Phase 2 must finish** before Phase 3 can run end-to-end (P_T→A weights are loaded in Stage 2).
- **Phase 3 freeze code** before Phase 4 launches multi-GPU runs (no live edits during training).

## Backwards Compatibility

Additive: no edits to `span_utils.py`, `span_fdd_finetune.py`, `span_finetune.py`. Existing scripts continue to work. Only `arguments.py` gets additive args (no rename, no default changes).

## Success Criteria (project-level)

1. SpanResidual avg Rouge-L > Residual baseline (γ=0) on both setups (primary).
2. SpanResidual > SpanFDD on both setups (secondary).
3. Reproduce Residual baseline within ±0.5 Rouge-L of paper Table 1 row "Ours" (sanity).
4. γ-sweep ablation shows non-zero γ optimal.

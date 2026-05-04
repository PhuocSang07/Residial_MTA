---
title: "Hyperparameters — On et al. ICLR 2026 (Residual Learning KD)"
source: "On et al. - 2026 - KNOWLEDGE DISTILLATION FOR LARGE LANGUAGE MODELS THROUGH RESIDUAL LEARNING.html"
date: 2026-05-03
---

# Paper Hyperparameters — On et al. ICLR 2026

## Stage 1 — Projector Pretraining (P_T→A, P_A→T)

| Parameter | Paper value |
|-----------|-------------|
| Epochs | 10 |
| Learning rate | 1e-3 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Scheduler | Cosine decay |
| Warmup | 0 |
| Bottleneck dim d_A | 64 |
| Projector bias | False |
| Batch size (global) | not explicitly stated (Stage 2 = 128; Stage 1 likely similar) |
| Data | Dolly train split |
| Teacher precision | bfloat16 |
| Projector precision | bfloat16 (same as DS run) |

## Stage 2 — KD Fine-tuning

| Parameter | Paper value |
|-----------|-------------|
| Epochs | 10 |
| Learning rate (student) | 1e-3 |
| Learning rate (projectors P_S→A, P_A→S) | 1e-4 |
| Optimizer | AdamW |
| Weight decay | not specified (likely 1e-2 per standard) |
| Scheduler | Cosine |
| Global batch size | 128 |
| Max sequence length | 512 |
| Lambda λ (residual weight) | 0.5 |
| Precision | bfloat16 |

## Architecture

| Component | Paper value |
|-----------|-------------|
| Anchor dim d_A | 64 |
| Projector layers | Linear (no bias) |
| β formula | `sqrt(d_S/d_A) * (1/n) Σ ‖h^S_i‖ / ‖h^(T→A)→S_i‖` |
| β scope | Sequence-level, detached (no backprop) |

## LoRA (used for TinyLlama student only)

| Parameter | Paper value |
|-----------|-------------|
| Rank r | 256 |
| Alpha | 8 |
| Dropout | 0.1 |
| Modules | Q, O, Gate, Up, Down projections |

## Cross-Tokenizer Setup (paper Table 4)

| Setting | Value |
|---------|-------|
| Teacher | Mixtral-8×7B-Instruct |
| Student | GPT2-120M (full FT) |
| Dolly Rouge-L | 22.62 |

## Hardware

| Setting | Value |
|---------|-------|
| GPUs | 8 × A100 40GB |
| Framework | PyTorch Distributed |

## Evaluation

| Setting | Value |
|---------|-------|
| Dolly test set | 500 samples |
| Benchmarks | Dolly, Self-Instruct, VicunaEval, S-NI (sinst), UnNI (uinst) |
| Metric | Rouge-L |
| Decoding | Greedy (do_sample=False) |

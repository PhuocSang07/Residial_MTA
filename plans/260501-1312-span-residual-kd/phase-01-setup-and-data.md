# Phase 01 — Setup, Data Preprocessing, Eval Benchmarks

## Context Links

- Brainstorm spec: `plans/reports/brainstorm-260501-0958-span-residual-kd.md`
- Reference scripts: `MTA/distillm-master/scripts/gpt2/tools/process_data_dolly.sh`, `scripts/qwen1.5/tools/process_data_dolly.sh`
- Encoder: `MTA/distillm-master/tools/process_data_dolly.py`
- spaCy span extractor: `MTA/distillm-master/span_utils.py::get_spans_offsets`
- Phase 1 status report: `plans/reports/phase-01-status-260503.md`

## Overview

- **Priority:** P1 (blocker for all downstream phases)
- **Status:** completed (2026-05-03) — see status report; canary smoke deferred (non-blocking)
- **Description:** Verify environment, processed Dolly tensors for both tokenizers, and 5 eval benchmarks present. Pre-cache spaCy spans to disk (CPU bottleneck mitigation).

## mlj-exp State Snapshot (2026-05-03)

Verified before execution; most of Phase 1 already satisfied by `mlj-exp/` repo state:

- ✅ Env: torch 2.8.0 + CUDA 12.8, transformers 5.5.4, deepspeed 0.18.9, peft 0.19.1, spacy 3.8.14 (+ `en_core_web_sm`), 1× RTX PRO 6000 Blackwell 102 GB. Logged to `MTA/distillm-master/logs/env.txt`.
- ✅ Processed Dolly: `MTA/distillm-master/processed_data/dolly/full/{gpt2,qwen,mistral,opt,openllama2}/{train,valid}_0.{bin,idx,jsonl}` all present. Train=11435, valid=1000.
- ✅ Span pre-extract already on disk: `MTA/data/dolly/syntactic_parsing.jsonl` and `processed_data/dolly/full/{gpt2,qwen}/syntactic_parsing.jsonl` (11435 lines = 1:1 with train.jsonl). Schema: `{phrases_lvl1: [{label,text,start_char,end_char}], phrases_lvl2: [...]}`. **Replaces planned `spans_cache.pkl`** — Phase 3 trainer will load this JSONL instead of pickling a new cache.
- ✅ Raw eval benchmarks: `MTA/data/{self-inst,vicuna}/valid.jsonl` flat; `MTA/data/{sinst,uinst}/11_/valid.jsonl` nested under `11_/` subfolder (n-shot prefix). All non-empty.
- ✅ Symlinks so distillm-master scripts (`${BASE_PATH}=./distillm-master`) resolve eval data and checkpoints:
  - `MTA/distillm-master/data` → `../data`
  - `MTA/distillm-master/checkpoints` → `../ckpt`
- ✅ Pretrained checkpoint present: `MTA/ckpt/gpt2/gpt2-base/mta_dskd_v2_eta/forward_kl-bf16__teacher_qwen1.5__...` (one Qwen-teacher→GPT2 distilled run available; Phase 2/3 still need a frozen teacher checkpoint(s) — fetch in Phase 2 if absent).
- ⚠️ `sinst` / `uinst` nested under `11_/` — Phase 5 eval scripts must point `--data-dir data/sinst/11_` (not `data/sinst`). Documented for Phase 5.
- ⏭️ 50-iter SFT canary smoke: deferred — non-blocking; same data path already used by prior runs in `results/gpt2/train/`.

Net Phase 1 work for mlj-exp reduced to: env probe + 2 symlinks + documentation. Download/preprocess scripts not authored (unnecessary).

## Key Insights from Scouting

- Training scripts reference `processed_data/dolly/full/{gpt2,qwen}/` — directory **does not exist** on disk; must run preprocessing.
- `MTA/distillm-master/data/` directory **does not exist**; raw Dolly JSONL must be obtained (from MiniLLM data tarball per `distillm-master/README.md`).
- Eval datasets (SelfInst/VicunaEval/S-NI/UnNI) **not yet present** under `MTA/`. Must download from the same MiniLLM/distillm tarball before Phase 5.
- `span_utils.py::get_spans_offsets` runs spaCy with `n_process=4` per batch — at train time this re-runs every step. Pre-extracting at preprocessing time avoids ~25% epoch overhead.
- Existing `arguments.py` has all needed args; no edits in Phase 1.

## Requirements

**Functional:**
1. `MTA/distillm-master/data/dolly/{train,dev,valid}.jsonl` exists and is non-empty.
2. `MTA/distillm-master/processed_data/dolly/full/{gpt2,qwen}/` exists with `.bin/.idx` outputs.
3. Eval benchmarks under `MTA/distillm-master/data/{self-inst,vicuna,sinst,uinst}/` (or equivalent paths used by `evaluate_main.py`).
4. Optional: `MTA/distillm-master/processed_data/dolly/spans_cache.pkl` keyed by sample idx → (spans_offsets, words_offsets).

**Non-functional:**
- Reproducibility: pin spaCy `en_core_web_sm` version, log torch / transformers versions to `logs/env.txt`.
- Disk budget: processed Dolly ~200MB; spans cache ~30MB; eval datasets ~50MB total.

## Architecture

Data flow:
```
Raw JSONL (data/dolly/{train,dev}.jsonl)
   ↓ tools/process_data_dolly.py
Tokenized binaries (processed_data/dolly/full/{gpt2,qwen}/)
   ↓ tools/precompute_spans.py  (NEW; optional)
spans_cache.pkl  (per-sample noun_chunks + verb_phrases offsets)

Eval raw (data/{self-inst,vicuna,sinst,uinst}/{prompt,reference}.jsonl)
   — consumed directly by PromptDataset (no preprocessing required)
```

## Related Code Files

**Modify:** none.

**Create:**
- `MTA/distillm-master/tools/precompute_spans.py` — one-pass spaCy over tokenized Dolly samples, save `(spans_offsets, words_offsets)` keyed by sample idx. (~80 LOC.)
- `MTA/distillm-master/scripts/spanresidual/setup-env.sh` — pip installs (deepspeed, spaCy model, peft) + sanity probe. (~30 LOC.)
- `MTA/distillm-master/scripts/spanresidual/download-data.sh` — wget/curl MiniLLM data tarball, extract to `data/`. Documents fallback URLs. (~40 LOC.)
- `MTA/distillm-master/scripts/spanresidual/preprocess-dolly.sh` — wraps `tools/process_data_dolly.sh` for both gpt2 and qwen tokenizers. (~30 LOC.)

**Delete:** none.

## Implementation Steps

1. **Env probe.** Run `python -c "import torch, deepspeed, transformers, peft, spacy; print(...versions)"` from `MTA/distillm-master/`. Note all versions to `logs/env.txt`. Run `python -m spacy download en_core_web_sm` if missing.
2. **Acquire raw datasets.** Create `scripts/spanresidual/download-data.sh`:
   - Download MiniLLM data tarball (URL in `distillm-master/README.md`): Dolly + SelfInst + VicunaEval + S-NI + UnNI.
   - Extract to `MTA/distillm-master/data/`.
   - Verify expected JSONL files exist; abort with clear error if not.
3. **Preprocess Dolly for gpt2 tokenizer.** Run `bash scripts/gpt2/tools/process_data_dolly.sh ./distillm-master`. Produces `processed_data/dolly/full/gpt2/`.
4. **Preprocess Dolly for qwen tokenizer.** Run `bash scripts/qwen1.5/tools/process_data_dolly.sh ./distillm-master`. Produces `processed_data/dolly/full/qwen/`.
5. **Precompute spaCy spans (optional, recommended).** Create `tools/precompute_spans.py`:
   - Load tokenized Dolly train + dev binaries.
   - Decode each sample's input_ids (gpt2 tokenizer; spans depend on raw text only, tokenizer-agnostic).
   - Apply `get_spans_offsets(texts, nlp, matcher)` from `span_utils.py`.
   - Pickle dict `{sample_idx: (spans, words)}` to `processed_data/dolly/spans_cache.pkl`.
   - Add CLI flag in Phase 3 trainer to load cache; falls back to runtime extraction if absent.
6. **Verify eval benchmarks.** Confirm directory layout matches `evaluate_main.py` expectations. If structure differs from MiniLLM tarball, write thin adapter symlinks under `data/`.
7. **Smoke test.** Run `scripts/gpt2/sft/sft_base.sh` for 50 iters as canary — confirms data pipeline + DeepSpeed config correct before any SpanResidual code.

## Todo Checklist

- [x] Run env probe; capture versions → `logs/env.txt`
- [~] Write `scripts/spanresidual/download-data.sh` — SKIPPED (data already in repo)
- [x] Acquire raw Dolly + 4 eval datasets to `data/` — already present at `MTA/data/`
- [x] Run gpt2 Dolly preprocessing — already done (`processed_data/dolly/full/gpt2/`)
- [x] Run qwen Dolly preprocessing — already done (`processed_data/dolly/full/qwen/`)
- [~] Write `tools/precompute_spans.py` — SKIPPED (`syntactic_parsing.jsonl` already on disk; Phase 3 will load JSONL format directly)
- [x] Run span precompute → `syntactic_parsing.jsonl` (already exists, 11435 lines × 2 tokenizers)
- [x] Verify eval benchmark dirs match `PromptDataset` expectations — `data/{self-inst,vicuna}/valid.jsonl` flat, `data/{sinst,uinst}/11_/valid.jsonl` nested
- [x] Symlinks: `distillm-master/{data,checkpoints}` → `../{data,ckpt}`
- [ ] 50-iter SFT canary run — deferred (non-blocking)

## Success Criteria

- `processed_data/dolly/full/gpt2/{train,valid}.bin` exists and `wc -c > 0`.
- `processed_data/dolly/full/qwen/{train,valid}.bin` exists and `wc -c > 0`.
- All 5 eval data dirs accessible by `PromptDataset(args.data_dir, split="valid")`.
- 50-iter canary SFT prints decreasing loss without crash.
- `spans_cache.pkl` loadable via `pickle.load`; covers ≥99% of train samples.

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| MiniLLM tarball URL unreachable | M | H | Document alternate HuggingFace mirror; fall back to manual download |
| spaCy span precompute slow | M | M | `n_process=4`, expect ~10 min for 11k Dolly; if slower, downsample for ablation runs |
| qwen tokenizer mismatch in spans | L | M | Spans are char-offset based; tokenizer-agnostic — verified in `span_utils.py` |
| Disk full | L | H | Pre-flight `df -h`; total budget < 1GB |

## Rollback Plan

All artifacts under `data/` and `processed_data/` are derived from raw inputs — wipe and re-run. No downstream code edits in this phase.

## Next Steps / Dependencies

- **Unblocks:** Phase 2 (needs `processed_data/dolly/full/{gpt2,qwen}/`), Phase 4 (training), Phase 5 (eval).
- **Phase 2 imports:** `processed_data/dolly/full/*/`.

## Unresolved Questions

- None for this phase. Question Q6 from brainstorm is resolved here: eval datasets are acquired in step 2.

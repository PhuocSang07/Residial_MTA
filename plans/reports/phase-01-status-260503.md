---
phase: 01
date: 2026-05-03
work_context: /teamspace/studios/this_studio/mlj-exp/
status: completed
---

# Phase 1 Status — Setup & Data Verification (mlj-exp)

## Outcome

Phase 1 closed. mlj-exp repo already shipped with: processed Dolly tensors (5 tokenizers), pre-extracted span syntactic parsing JSONL, and raw eval benchmarks. Only env probe + 2 symlinks needed to make `MTA/distillm-master/` scripts find data/checkpoints.

## What was done

1. **Env probe** → `MTA/distillm-master/logs/env.txt`
   - python 3.12.11
   - torch 2.8.0 + cu128
   - transformers 5.5.4
   - deepspeed 0.18.9
   - peft 0.19.1
   - spacy 3.8.14 + `en_core_web_sm` loaded OK
   - numpy 1.26.4, datasets 4.8.4, accelerate 1.13.0, sentencepiece 0.2.1
   - GPU: 1× RTX PRO 6000 Blackwell, 102 GB VRAM (single-GPU comfortably fits both planned setups)

2. **Verified processed Dolly** at `MTA/distillm-master/processed_data/dolly/full/`:
   - 5 tokenizer dirs: gpt2, qwen, mistral, opt, openllama2 (Phase 2-3 use gpt2 + qwen only)
   - per dir: `train_0.bin`, `train_0.idx`, `valid_0.bin`, `valid_0.idx`, `train.jsonl`, `valid.jsonl`
   - sample counts: train=11435, valid=1000 (matches Dolly 11k+1k convention)
   - gpt2/qwen each also has `syntactic_parsing.jsonl` (11435 lines)

3. **Symlinks created** in `MTA/distillm-master/`:
   - `data` → `../data`
   - `checkpoints` → `../ckpt`
   - Now `${BASE_PATH}=./distillm-master` scripts (BASE_PATH passed as arg 1) resolve `${BASE_PATH}/data/...` and `${BASE_PATH}/checkpoints/...` correctly.

4. **Eval benchmarks verified** under `MTA/data/`:
   - `dolly/{train,dev,valid}.jsonl` — present (additionally `syntactic_parsing.jsonl`)
   - `self-inst/valid.jsonl`, `vicuna/valid.jsonl` — flat
   - `sinst/11_/valid.jsonl`, `uinst/11_/valid.jsonl` — **nested under `11_/`**
   - Bonus dirs: `dialog/`, `dpo/`, `gpt3_answers/`, top-level `dolly_train.jsonl`

## Key findings vs original plan

- `syntactic_parsing.jsonl` (already on disk) replaces planned `spans_cache.pkl`. Format is per-sample `{phrases_lvl1: [{label,text,start_char,end_char}], phrases_lvl2: [...]}`. **Phase 3 trainer must load this JSONL by sample idx instead of building a pickle cache.**
- `sinst` / `uinst` nest under `11_/` (looks like 1-shot prefix from MiniLLM tarball). Phase 5 eval `--data-dir` must be `data/sinst/11_` not `data/sinst`.
- `ckpt/gpt2/gpt2-base/mta_dskd_v2_eta/...` already has one prior distill run; Phase 2 stage-1 needs a frozen GPT2-1.5B teacher SFT checkpoint, not yet confirmed present — Phase 2 starts with locating/downloading it.
- Items skipped (intentionally): `download-data.sh`, `precompute_spans.py`, `preprocess-dolly.sh`, `setup-env.sh`. Repo already shipped these outputs.

## Files touched

- Created: `MTA/distillm-master/data` (symlink → `../data`)
- Created: `MTA/distillm-master/checkpoints` (symlink → `../ckpt`)
- Created: `MTA/distillm-master/logs/env.txt`
- Updated: `plans/260501-1312-span-residual-kd/phase-01-setup-and-data.md` (state snapshot + checklist)
- Updated: `plans/260501-1312-span-residual-kd/plan.md` (status row)

## Deferred / out of scope

- 50-iter SFT canary smoke run (compute, ~10 min). Not blocking Phase 2; same paths already used by prior `results/gpt2/train/` artifacts. Run on demand when Phase 2 starts.

## Unresolved Questions for Phase 2

1. Where is the **frozen teacher SFT checkpoint** for GPT2-1.5B (paper baseline) and Qwen1.5-1.8B? Is `VoCuc/Qwen1.5_1.8B_SFT_Dolly` (referenced in earlier sessions) the canonical Qwen teacher? Need explicit teacher path before stage-1 projector pretrain can run.
2. Confirm whether `syntactic_parsing.jsonl` covers the `valid` split too — only `train.jsonl` was checked at 11435 lines. If valid split lacks pre-extracted spans, Phase 3 will fall back to runtime spaCy on the 1000 valid samples (fast).

**Status:** DONE
**Summary:** Env + data layout verified for mlj-exp; symlinks added; phase-01 doc updated; key finding (syntactic_parsing.jsonl replaces spans_cache.pkl) documented for Phase 3.
**Concerns/Blockers:** Teacher checkpoint location for Phase 2 not yet pinned down.

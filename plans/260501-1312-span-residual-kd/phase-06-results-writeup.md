# Phase 06 — Results Writeup (optional)

## Context Links

- Aggregated results: `eval_outputs/results_table.md` (Phase 5)
- γ ablation: `eval_outputs/ablation_gamma.md` (Phase 5)
- Brainstorm hypothesis + spec: `plans/reports/brainstorm-260501-0958-span-residual-kd.md`

## Overview

- **Priority:** P3 (optional)
- **Status:** pending
- **Description:** Distill Phase 5 outputs into a paper-ready writeup section: tables, plots, narrative on hypothesis verification.

## Requirements

**Functional:**
1. Final results table (mean ± std) for 4 baselines × 5 benchmarks × 2 setups.
2. γ-sweep plot (line chart: γ on x-axis, Rouge-L on y-axis).
3. Loss-magnitude plot from Phase 3 first-100-step logs (validates loss balance).
4. Narrative: hypothesis verified vs not, attribution analysis (which component contributes how much).
5. Limitations + future work section (cross-tokenizer, MoE expert fusion noted as out of scope).

**Non-functional:**
- Plain Markdown for now; optional LaTeX export.

## Architecture

Inputs: Phase 5 aggregated tables + per-step training logs.
Output: `plans/260501-1312-span-residual-kd/results.md` plus generated PNG plots under `visuals/`.

## Related Code Files

**Modify:** none.

**Create:**
- `plans/260501-1312-span-residual-kd/results.md` — narrative + tables.
- `MTA/distillm-master/tools/plot_gamma_sweep.py` — matplotlib line plot from `ablation_gamma.md` JSON. (~60 LOC.)
- `MTA/distillm-master/tools/plot_loss_magnitudes.py` — parse `log.txt` lines for `L_SFT/L_res/L_span` and plot first 100 steps. (~50 LOC.)

## Implementation Steps

1. Parse final `results_table.md`; copy into `results.md` with bold-best-per-row.
2. Run plotting scripts → save PNG to `visuals/`.
3. Author narrative:
   - Recap hypothesis (span ⊥ residual).
   - Empirical verdict per setup.
   - Attribution: SpanResidual − Residual γ=0 = span contribution; SpanResidual − SpanFDD = residual contribution; intersection inferred.
   - Failure modes if any.
4. Limitations:
   - Same-tokenizer only.
   - No PEFT/LoRA tested.
   - Single dataset (Dolly) for projector pretrain.
5. Future work:
   - Cross-tokenizer P_A→S generalization.
   - SpanResidual + DistiLLM-2.
   - MoE expert routing on residual mask.
6. Visual aids via `/ck:preview --diagram <topic>` for architecture illustration if needed.

## Todo Checklist

- [ ] Author narrative `results.md`
- [ ] Embed final table
- [ ] Generate γ-sweep plot
- [ ] Generate loss-magnitude plot
- [ ] Document limitations + future work
- [ ] Cross-link to brainstorm + plan docs

## Success Criteria

- `results.md` < 300 lines, self-contained, includes all 5 success criteria from `plan.md`.
- Plots are readable (≥ 150 DPI, axis labels, legends).
- Verdict on hypothesis is unambiguous.

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Hypothesis falsified | M | L | Document neutrally; analysis still publishable as negative result |
| Plot generation fails | L | L | Use fallback Markdown ASCII tables |

## Rollback Plan

Pure documentation; no rollback needed.

## Next Steps / Dependencies

- **Blocked by:** Phase 5 (aggregated results).
- **Unblocks:** Final delivery / paper draft.

## Unresolved Questions

- None.

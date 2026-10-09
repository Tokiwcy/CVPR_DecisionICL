# Audited controls and what they test

All controls below use Qwen3-VL-4B BF16 and the corresponding frozen test set. They are separate variants with their own specified selection/calibration contract; no test outcome was used to choose weights or factors. Read paired rescue/harm, bootstrap 95% CI, and unadjusted two-sided exact McNemar p for **each** comparison in [paired_results.csv](paired_results.csv). The comparisons share test examples and are not independent trials.

## Remove factor evidence; keep same-class selection

For 2-shot RICES on the complete DTD, Aircraft, and CUB tests, the no-factor gate chooses the class of the parent retriever's top-ranked example, then takes the top two RICES examples from that class in the complete training pool. It has no learned factor weights. This directly asks whether same-class grouping alone explains DecisionICL's gain.

| Dataset | RICES | No-factor class gate | DecisionICL | No-factor−DecisionICL |
| --- | ---: | ---: | ---: | ---: |
| DTD | 77.87 | 76.91 | 78.78 | −1.86 pp |
| Aircraft | 59.20 | 56.56 | 65.71 | −9.15 pp |
| CUB | 79.10 | 79.49 | 80.77 | −1.27 pp |

All three no-factor comparisons are below DecisionICL. This is evidence that the particular factor-guided class decision contributes beyond this simple same-class rule on these three datasets; it does not show that every possible same-class selector is inferior.

## Rotate the factor/class mapping

The fixed one-step class-prototype rotation breaks the intended factor-to-class correspondence while preserving the 2-shot RICES setting. Rotated−DecisionICL is −2.07 pp on DTD, −9.09 pp on Aircraft, and −1.27 pp on CUB. CUB's validation fit fell back to the parent-only endpoint, so its result should not be described as an isolated penalty from wrong factor semantics.

## Generic direct-sample alternative

Across 25 Qwen3 cells, the Generic alternative is below DecisionICL in 24 cells, with an equal-cell Generic−DecisionICL mean of −1.36 pp. This alternative scores generic visual descriptions directly against candidate samples and also changes retrieval/calibration, so it is a broader method comparison, **not** a controlled factor-vocabulary-only test. The [matched same-architecture 1-shot factor-source comparison](matched_source_control.json) that already exists covers only DTD and Aircraft × RICES/MUIER: DecisionICL−Generic is −0.37, +0.53, +1.14, and +0.69 pp, respectively. These four results are mixed and do not warrant a universal source-superiority claim.

## Frozen-CLIP visual class centroid

This component replacement uses frozen CLIP training-class visual centroids as class evidence, keeping the 2-shot RICES scaffold on DTD/Aircraft/CUB. Centroid−DecisionICL is +0.43, −1.89, and +5.05 pp, respectively. The result is mixed and the centroid wins substantially on CUB. It is a useful comparator to report honestly, not a proof that factor evidence always dominates generic visual class evidence. It is not scheduled for expansion.

The exact method outputs, invalid counts, paired rescue/harm, intervals and p-values are in [supplementary_results.md](supplementary_results.md). The source code/protocol mapping is in [research_code/README.md](../research_code/README.md).

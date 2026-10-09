# Remaining research work as of the exported record

The only authorized unfinished GPU experiment is the full Qwen3-VL-4B BF16 three-seed repeat. Its queue status is **paused_by_user**. Nothing in this file instructs an automatic restart, and partial raw/cache files are not paper results.

## Fixed repeat scope

The audited reference is seed **42**. Fresh seeds **17** and **73** cover DTD, Aircraft, CUB, Stanford Dogs, and Flowers102, each at 1 and 2 shots. Every new seed/dataset/shot block runs all five 1-shot methods (Random, RICES, MUIER, RICES+DecisionICL, MUIER+DecisionICL) or seven 2-shot methods (those five plus CIRCLES and CIRCLES+DecisionICL). That is `2 seeds × 5 datasets × (5+7 methods) = 120` additional method results. Target-model raw batches, predictions, and caches from seed 42 are **not** reused. Seed-dependent random demonstrations and validation folds change; the target model, factor bank, frozen visual evidence, CLIP encoder, prompts, and greedy decoding remain fixed. DecisionICL calibration is refit for each seed/dataset/shot using its own train/validation contract; test labels are not tuning inputs.

## Audited and remaining blocks

| Seed | Dataset | 1-shot | 2-shot | New method items still pending |
| ---: | --- | --- | --- | ---: |
| 17 | DTD | 5/5 audited | 7/7 audited | 0 |
| 17 | Aircraft | 5/5 audited | 7/7 audited | 0 |
| 17 | CUB | 5/5 audited | 7/7 audited | 0 |
| 17 | Dogs | 5/5 audited | 0/7 audited; partial inference cache | 7 |
| 17 | Flowers102 | 0/5 | 0/7 | 12 |
| 73 | DTD | 0/5 | 0/7 | 12 |
| 73 | Aircraft | 0/5 | 0/7 | 12 |
| 73 | CUB | 0/5 | 0/7 | 12 |
| 73 | Dogs | 0/5 | 0/7 | 12 |
| 73 | Flowers102 | 0/5 | 0/7 | 12 |

Total: **41/120 independently audited, 79/120 pending**. The seed-17 Dogs 2-shot isolated root was safely paused at **27,113 cache rows**, comprising **13,556 complete fixed batch-2 groups plus one first-row orphan**. The orphan is not counted as a complete raw batch or method result. The existing isolated root and pause/hash receipts must be preserved. Before any user-authorized continuation, independently verify the partial cache and plan, establish a new isolated root and a single-GPU lock, and resume only through an audited continuation wrapper. Do not edit or blindly restart the frozen scientific runner or paused root in place.

For each newly completed block, the independent audit must confirm complete/run/plan hashes, full query/method coverage, fixed raw-batch partners and strict parsing, target-cache provenance, accuracy/macro/invalid, and paired rescue/harm with bootstrap 95% CI and raw two-sided exact McNemar p. Only after that may the result ledger and paper tables be updated. Three-seed summaries must wait until all 120 new items are audited; the 41 interim rows must not be cherry-picked into the paper main table.

## Other manuscript work, not queued inference

- Reconcile the completed no-factor, rotated, Generic and centroid controls with the central claim and show the heterogeneous per-dataset gains, invalid-output rates, rescue/harm and confidence intervals. The source-only four-cell comparison is mixed; do not claim that target-model factor wording alone is universally superior.
- Add a qualitative error/rescue panel and a runtime/compute accounting from **existing** audited logs/caches, with traceable example IDs and no post-test method change. These are analysis/presentation tasks, not new test tuning.
- Complete final CVPR-format manuscript, bibliography, reproducibility/limitations text and anonymity checks. No additional GPU ablation or extra shot/model sweep is currently authorized or scheduled. Any new factor-source-only experiment would need its datasets and calibration protocol fixed in advance and separate authorization.

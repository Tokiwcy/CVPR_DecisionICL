# DecisionICL

**DecisionICL** aligns in-context demonstration retrieval with visual decision factors elicited from the target vision-language model. The model proposes reusable factors; a frozen batched visual encoder measures factor evidence; validation-only calibration combines that evidence with a parent retriever. The method selects a class and then retrieves demonstrations directly from the **complete training pool** within that class. It is not a reranker over a small RICES candidate list.

This repository is an in-progress CVPR research release. It separates independently audited results from an authorized but **paused** multi-seed repeat. All accuracy and paired statistics below come from the frozen single-seed main experiments, not from unfinished repeats.

## Audited results intended for the paper

The main matrix covers Qwen3-VL-4B BF16, Qwen2.5-VL-7B NF4, and Gemma4 12B NF4 on DTD, Aircraft, CUB, Stanford Dogs, and Flowers102 at 1 and 2 shots: **180/180 method results independently audited**. It includes Random, RICES, MUIER, and (at 2 shots) CIRCLES, with a matched DecisionICL extension for each retrieval parent. Across the 75 matched parent→DecisionICL comparisons, 75 gains are positive. This is a descriptive count over correlated cells, not 75 independent tests.

| Target model | Equal-cell mean DecisionICL−parent gain | Audited matched cells |
| --- | ---: | ---: |
| Qwen3-VL-4B BF16 | +2.06 percentage points | 25 |
| Qwen2.5-VL-7B NF4 | +3.11 percentage points | 25 |
| Gemma4 12B NF4 | +3.52 percentage points | 25 |

The full 15-row model/dataset table, including each parent and its DecisionICL extension at each shot, is in [results/main_results.md](results/main_results.md). [results/supplementary_results.md](results/supplementary_results.md) gives every method cell and paired comparison. Machine-readable [method results](results/method_results.csv) and [paired results](results/paired_results.csv) retain full precision, macro accuracy, invalid outputs, rescue/harm, bootstrap 95% confidence intervals, and unadjusted two-sided exact McNemar p-values. These files keep the experiment's internal method ID in `source_method` while displaying **DecisionICL** as the publication name.

Collaborators can download the original artifacts and independently check the tables. The [artifact bundle guide](artifact_bundle/README.md) links the review ZIP included in the repository and the complete raw/cache/numeric evidence in the [GitHub evidence release](https://github.com/Tokiwcy/CVPR_DecisionICL/releases/tag/decisionicl-evidence-20261009). Run `python tools/recompute_artifact_results.py --review-archive artifact_bundle/decisionicl-review-artifacts.zip` to recompute the reported accuracy/macro/invalid counts and paired rescue/harm/exact p directly from the original predictions and test labels, without a GPU.

Audited Qwen3 controls are also included: 25 Generic direct-sample cells, a 3-dataset rotated factor/class control, a 3-dataset no-factor same-class gate, and a 3-dataset frozen-CLIP visual-centroid gate. Their different scientific questions and mixed findings are stated in [results/ablations.md](results/ablations.md). The Generic direct-sample variant changes the retrieval rule, so it does not isolate factor vocabulary by itself. Test outcomes were not used to choose factors, weights, prompts, or calibration parameters.

## Work still to run or finish

The **Qwen3 full three-seed repeat is paused by the user**, not complete. Seed 42 is the audited reference. Fresh seed 17 and 73 runs were planned over all five datasets, both shots, and every method, with no reuse of old target predictions: **120 new method results**. Only **41/120** have passed independent audit. Seed 17 is complete for DTD, Aircraft, and CUB at both shots and Dogs at 1 shot. Dogs 2-shot has partial raw cache only, and Flowers102 1/2-shot and every seed-73 block remain. There is **no valid three-seed aggregate yet**. See [results/pending.md](results/pending.md) for the exact remaining block list, cache boundary, audit gates, and the distinction between queued inference and optional future analyses. The paused run should be resumed only with a new isolated, independently preflighted root, not by editing the frozen runner or the old root in place.

No further GPU ablation is currently queued. The completed no-factor and other controls should be interpreted as they are; optional manuscript analyses and any additional experiments are listed separately in [results/pending.md](results/pending.md).

## Code and provenance

- [research_code/decisionicl.py](research_code/decisionicl.py) is a small, NumPy-only reference for the central class-prototype, calibration, and complete-pool selection operations. It is illustrative; it does not reproduce the full target-model pipeline by itself.
- [research_code/frozen/](research_code/frozen/) contains the unchanged research-source snapshot needed to trace the audited main and control experiments. Its [SHA-256 manifest](research_code/frozen/snapshot_manifest.json) records every copied file. Internal experiment identifiers remain intact to preserve the hash-bound protocols.
- [research_code/README.md](research_code/README.md) maps each result family to its actual runner, calibration, audit, and protocol files and explains reproducibility prerequisites.
- [results/export_manifest.json](results/export_manifest.json) records 152 verified source-file hashes, ledger inputs, exported-file hashes, and result counts. Those source files are now in the [review artifact ZIP](artifact_bundle/decisionicl-review-artifacts.zip). Run `python tools/verify_results.py` to check the published bundle. The author-side exporter is [tools/export_from_workspace.py](tools/export_from_workspace.py); collaborators can use the portable artifact verifiers to inspect and recompute the released summaries.

The pre-existing `src/behavior_icl/`, `src/decision_atoms/`, root `scripts/`, configurations, and tests belong to an earlier BehaviorICL spurious-dataset project. They were **not changed** for this release. Its original README is preserved as [LEGACY_BEHAVIORICL.md](LEGACY_BEHAVIORICL.md). DecisionICL paper numbers must be taken from `results/`, not from the legacy Waterbirds/Spawrious tables.

## Reproduction boundary

The [evidence release](artifact_bundle/README.md) provides original predictions, test-label manifests, audits, plans, calibration and referenced raw/cache/numeric resources. The portable verifier recomputes the table statistics from those files. Original GPU inference additionally requires dataset images, model weights and the matching environment/input contracts documented in the frozen protocols; dataset images and model checkpoints are not redistributed. The NumPy reference illustrates the selection rule and does not generate the paper statistics from synthetic data. For a lightweight code check, install NumPy and pytest, then run `python -m pytest tests/test_decisionicl.py -q` from `research_code/`.

Source hashes and calibration records support reproducibility of the reported runs. The paused multi-seed repeat is explicitly outside the current paper result table.

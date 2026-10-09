# DecisionICL research-code map

The paper-facing name is **DecisionICL**. The frozen experiment files retain internal names such as `atom_class_equal_raw` and `+Decision` because the independent audit and source hashes identify those exact bytes. Renaming those identifiers inside the scientific runners would break provenance. The top-level [README](../README.md), result tables, and [NumPy reference](decisionicl.py) use DecisionICL consistently.

## Core method

The target VLM discovers task-level visible decision factors from training material. A frozen batched visual encoder scores factor states for images. Training-class factor-state prototypes and a parent retriever produce per-query class features; each variant's coefficients are calibrated independently on its train/validation split. The chosen class gates **direct top-k retrieval over the complete training pool**, using the same parent similarity within that class. A changed factor bank, evidence scorer, fusion rule, model, or calibration split requires fresh validation calibration; no test labels enter selection or tuning.

The original factor/prototype/score implementation lives in [frozen/src/dgdr/](frozen/src/dgdr/) (`discovery.py`, `evidence.py`, `atom_label.py`, `calibration.py`, `retrieval.py`, `plans.py`). [decisionicl.py](decisionicl.py) illustrates the class-prototype and full-pool retrieval steps with only NumPy. It covers a simplified equal-weight class-feature contract; **the frozen runners, not this reference, produced the published predictions**. The original RICES, MUIER, and CIRCLES feature constructions differ, so do not treat one illustrative function as an executable replacement for every parent.

## Result-to-code mapping

| Released result family | Frozen runner / calibration | Independent audit and protocol |
| --- | --- | --- |
| Qwen3 five-dataset 1/2-shot main table | `frozen/scripts/run_qwen3_five_dataset_recalibrated.py`, `run_qwen3_circles_official_prompt_2shot_four.py`, `run_qwen3_flowers_replacement_queue.py`, `run_qwen3_flowers_auto_batch.py` | `frozen/protocols/qwen3_five_dataset_recalibrated_20260910_zh.md`; source-ledger SHA closure |
| Qwen2.5 NF4 five-dataset main table | `frozen/scripts/run_qwen25_nf4_five_dataset.py`, `run_qwen25_nf4_class_equal.py`, `qwen25_nf4_official_backend.py` | `frozen/protocols/qwen25_nf4_five_dataset_20260913_zh.md`; source-ledger SHA closure |
| Gemma4 12B NF4 five-dataset main table | `frozen/scripts/run_gemma12_main_eval_v7_20260928.py` plus isolated resume wrappers v8–v12, `calibrate_gemma12_main_v3_20260926.py` | `frozen/scripts/audit_gemma12_main_eval_v5_20260926.py`, `independent_recheck_gemma12_v9_20260930.py` through v12; `frozen/protocols/gemma12_main_1_2shot_v3_20260926_zh.md` |
| Generic direct-sample comparison | `frozen/scripts/run_qwen3_generic_direct_1_2shot_20260921.py` | `frozen/protocols/qwen3_generic_direct_1_2shot_20260921_zh.md`; source-ledger SHA closure |
| Rotated factor/class mapping | `frozen/scripts/run_qwen3_rotated_factor_class_2shot_20261003.py` | Matching `audit_*.py` and protocol in `frozen/protocols/` |
| No-factor same-class gate | `frozen/scripts/run_qwen3_parent_class_gate_2shot_20261004.py` | Matching `audit_*.py` and protocol in `frozen/protocols/` |
| Frozen-CLIP visual centroid gate | `frozen/scripts/run_qwen3_visual_centroid_class_gate_2shot_20261004.py` | Matching `audit_*.py` and protocol in `frozen/protocols/` |
| Paused Qwen3 seed-17/73 repeat | `frozen/scripts/run_qwen3_main_three_seed_full_20261004.py`; isolated resume wrappers in `frozen/resume_scripts/` | `frozen/scripts/audit_qwen3_main_three_seed_block_20261005.py`, `frozen/protocols/qwen3_main_three_seed_full_20261004_zh.md`; only 41/120 audited |
| Paper ledger/table construction | `frozen/analysis/build_experiment_results_ledger.py`, `frozen/analysis/build_tables.py` | [result export manifest](../results/export_manifest.json) and `tools/verify_results.py` |

The [snapshot manifest](frozen/snapshot_manifest.json) gives the SHA-256 of each copied scientific source/protocol file. This is a **code snapshot**, not a complete artifact mirror: model checkpoints, licensed datasets, train/validation/test manifests, factor banks, frozen visual evidence, embeddings, target raw batches, audit receipts, and full source-hash closure remain in the original research workspace. Some original scripts use workspace-relative artifact paths. Reproducing the GPU results requires restoring matching inputs and checking their hashes; simply invoking a frozen runner from this repository will not regenerate the published numbers.

## Lightweight check

With NumPy and pytest installed, run from this directory:

```bash
python -m pytest tests/test_decisionicl.py -q
```

That tests class-prototype probabilities, validation-only calibration behavior, and retrieval from the complete class pool. It does not validate target-model accuracies. For the published tables, run `python tools/verify_results.py` at the repository root; it checks export hashes and numerical consistency without private artifacts.

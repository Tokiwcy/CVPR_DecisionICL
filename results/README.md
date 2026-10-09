# DecisionICL result bundle

This directory is the paper-facing export of independently audited experiment records. `DecisionICL` is the publication name; the `source_method` and `source_comparator` columns preserve the frozen internal identifiers. Percentages in Markdown are rounded; CSV accuracies are fractions in `[0, 1]`, while paired deltas and confidence limits are **percentage points**.

| File | Purpose |
| --- | --- |
| [main_results.md](main_results.md) | Complete 3-model × 5-dataset × 1/2-shot main table and compact means |
| [supplementary_results.md](supplementary_results.md) | All 214 audited method rows and 149 paired rows |
| [method_results.csv](method_results.csv) | Full-precision accuracy, macro accuracy and invalid output counts |
| [paired_results.csv](paired_results.csv) | Full-precision paired gain, rescue/harm, bootstrap 95% CI and raw two-sided exact McNemar p |
| [matched_source_control.json](matched_source_control.json) | Four audited same-architecture 1-shot DecisionICL/Generic factor-source comparisons |
| [ablations.md](ablations.md) | Scientific question, measured result, and limitations for each control |
| [pending.md](pending.md) | Detailed paused three-seed schedule and remaining work |
| [paper_numbers.json](paper_numbers.json) | Cross-table counts and compact descriptive statistics |
| [source_manifest.json](source_manifest.json) | SHA-256 identities of 152 source evidence files |
| [export_manifest.json](export_manifest.json) | Input/output hashes and export counts |

The main 180 result rows are complete. The other 34 rows are the 25-cell Qwen3 Generic direct-sample comparison and three 3-cell Qwen3 controls. The paused seed-17/73 repeat has 41 independently audited **additional** method rows, but these are not included in `method_results.csv` or the single-seed main table because the repeat is incomplete. A full three-seed mean, standard deviation, or test statistic has not been computed.

The result exporter checked each recorded source file SHA before writing these summaries. `python tools/verify_results.py` verifies exported hashes, row counts, the main matched-pair matrix and numerical consistency without the original model or datasets. Complete experiment reproduction needs the frozen code and inputs described in [research_code/README.md](../research_code/README.md).

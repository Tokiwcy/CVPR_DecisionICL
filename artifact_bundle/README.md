# DecisionICL artifacts for collaborators

The repository includes a small [review archive](decisionicl-review-artifacts.zip) containing the original result sources, block receipts, predictions, test-label manifests, frozen plans, and calibration records. The full raw/cache/numeric evidence is available as assets in the [evidence release](https://github.com/Tokiwcy/CVPR_DecisionICL/releases/tag/decisionicl-evidence-20261009). Download links, sizes, and SHA-256 digests are recorded in [release.json](release.json).

The review ZIP is **16.9 MB / 385 original files**. The full release is approximately **1.64 GB**, representing **124,910 original logical paths** stored as **121,694 unique file objects**, including one preserved historical cache version. Exact counts and byte sizes are in [summary.json](summary.json).

## Quick table review

From the repository root, with ordinary Python and no model/GPU dependencies:

```bash
python tools/verify_results.py
python tools/recompute_artifact_results.py --review-archive artifact_bundle/decisionicl-review-artifacts.zip
```

The first command checks publication/export/code hashes and each review-archive member. The second reads original predictions and test labels to independently recompute all **214** reported method accuracies, macro accuracies and invalid-output counts, plus all **149** paired deltas, rescue/harm counts and unadjusted two-sided exact McNemar p-values, and the four older matched factor-source comparisons. Bootstrap confidence intervals remain in the original audited receipts; this lightweight command does not replace them with a new random draw.

To inspect files, unzip `decisionicl-review-artifacts.zip` into an empty directory. It retains the original workspace-relative paths. [review_manifest.json](review_manifest.json) records each included file's original SHA-256. [evaluation_index.json](evaluation_index.json) maps every public method cell to its exact prediction file, internal method identifier, and test-label manifest.

## Full evidence and raw provenance

```bash
python tools/verify_artifact_bundle.py --download --bundle downloaded_evidence
python tools/recompute_artifact_results.py --bundle downloaded_evidence
```

The full package includes the dependency closure of the original ledger sources, the completed main/control blocks, four older matched factor-source comparisons, and the 41 audited interim seed-repeat method items. It includes target raw batches, cache files, discovery/description records where referenced, calibration, frozen plans, image/label metadata, frozen numeric evidence/embeddings and original source/protocol dependencies. The unfinished seed-repeat cache is not a complete result and is outside the reported paper table.

Files shared between isolated resume roots are stored once under `objects/<sha256>` in the ZIP assets. `decisionicl-artifact-manifest.json.gz` maps every original logical path to the right archive and object. The verifier checks archive digests and individual file/object hashes on any operating system. To reconstruct original directories as readable files:

```bash
python tools/verify_artifact_bundle.py --bundle downloaded_evidence --materialize evidence_workspace
```

Use a new or empty destination. Original JSON bytes are preserved, including historical absolute paths such as `D:\BehaviorICL`; the portable verifiers resolve paths through the release index rather than requiring that Windows drive. Original frozen audit runners can require their original workspace layout and dependencies.

An early preflight recorded a partial validation-description cache before it was appended. The full bundle preserves the final cache and a separate historical object recovered from an **exact SHA-matching byte prefix**, with its original hash and recovery information in `historical_versions`. Materialization writes `HISTORICAL_VERSIONS.json`; it does not overwrite the final cache with the older version.

Dataset images and model checkpoints remain external inputs. The package therefore supports inspecting/recomputing the reported evaluation and checking the released evidence; reproducing target-model inference still requires the original datasets and model weights. The manifest lists explicitly excluded file-hash references. No experiment is restarted by these verification commands.

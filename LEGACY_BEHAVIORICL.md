# BehaviorICL

BehaviorICL studies which demonstrations teach a frozen vision-language model the intended visual rule in in-context classification.

The main setting is 3-shot visual ICL with Qwen3-VL-4B-Instruct in native bfloat16 and SigLIP2 image similarity. Standard RICES uses global top-k retrieval: there is no class quota and the most similar selected demonstration is placed closest to the query.

## Methods

### 1. RICES baseline

For query image `q` and training demonstration `d`, RICES selects the global top-k examples by cosine similarity:

```text
score(d, q) = similarity(d, q)
```

### 2. Oracle diagnostic selection

This controlled method uses **training-set spurious-attribute metadata**. It never uses query/test metadata for selection.

For Waterbirds, a training example is diagnostic when bird label and background disagree. For Spawrious O2O, it is diagnostic when a breed appears in the shared filler location instead of its aligned location.

Two selectors are included:

- `hard_coverage_global_3shot`: retain RICES unless the context has no diagnostic example; then replace the least-similar selected demo with the closest diagnostic demo.
- `weighted_*_global_3shot`: select global top-k using

```text
score(d, q) = similarity(d, q) + lambda * diagnostic(d)
```

`lambda` is selected separately for each dataset/model/shot setting on validation and frozen before test evaluation.

### 3. Metadata-free failure-contrastive discovery

This method asks what a demonstration/rule teaches rather than whether an individual image is merely difficult.

1. Run RICES on validation and mine frequent `(gold label -> predicted label)` confusion directions.
2. In four out-of-fold splits, construct each induction panel from representative errors, visually matched correct gold controls, and visually matched correct confuser controls.
3. Ask the frozen VLM to induce visible class cues, shared non-diagnostic cues, a failure hypothesis, and a corrective rule.
4. Keep the original RICES demonstrations and append the learned rule only to queries routed by the baseline prediction.
5. Enable a confusion pair only when its held-out utility is positive:

```text
U = rescue - beta * harm
```

`rescue` counts baseline errors corrected by the rule; `harm` counts baseline-correct predictions changed to errors. The released safety setting is `beta=2`. Discovery and routing use labels, model predictions, and visual similarity, but no background/group annotations. Test labels and group metadata are read only after predictions are frozen for reporting.

## Main frozen results

Qwen3-VL-4B-Instruct, bfloat16, global 3-shot:

| Dataset | Method | Test accuracy | Worst-group accuracy |
|---|---:|---:|---:|
| Waterbirds | RICES | 85.00 | 74.80 |
| Waterbirds | hard diagnostic coverage | 85.70 | 75.60 |
| Waterbirds | weighted diagnostic, `lambda=0.075` | **87.80** | **80.40** |
| Spawrious O2O Easy | RICES | 89.58 | 79.27 |
| Spawrious O2O Easy | hard diagnostic coverage | 92.61 | 85.28 |
| Spawrious O2O Easy | weighted diagnostic, `lambda=0.20` | **94.75** | **86.83** |

Metadata-free failure-contrastive discovery:

- Spawrious O2O Easy, `beta=2`: 89.58% -> **91.78%**, with 576 rescues and 28 harms. Only Labrador<->Bulldog passed the all-fold validation gate. Worst-group accuracy remained 79.27% because the Dachshund direction was rejected.
- Waterbirds initial `beta=1` test was a negative result: 85.9% -> 86.0%, with 31 rescues and 30 harms, while worst-group accuracy fell from 71.6% to 62.8%. A later `beta=2` validation rule abstains, but this revision was made after inspecting that test and is **not** presented as an independent test result.

The failure-contrastive runs use separately frozen RICES prediction caches, so their RICES numbers should be compared only within that experiment, not substituted for the selection-table baselines above.

Machine-readable summaries are under [`results/`](results/).

## Repository layout

```text
configs/                         reproducible model/task templates
scripts/prepare_waterbirds.py    Waterbirds metadata -> JSONL manifest
scripts/prepare_spurious_tasks.py Spawrious/SpuCoAnimals/MetaShift preparation
scripts/run_mvp.py               controlled A/B/C/D Waterbirds experiment
scripts/run_three_shot_weighted.py Waterbirds RICES/hard/weighted runner
scripts/run_multiclass_weighted.py generic RICES/hard/weighted runner
scripts/run_failure_contrastive_discovery.py automatic rule discovery and gate
src/behavior_icl/                minimal Qwen scoring backend
src/decision_atoms/              manifest, SigLIP and target-model utilities
tests/                           selection, protocol and leakage tests
```

## Installation

Python 3.10+ and a CUDA environment compatible with PyTorch are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. NF4 experiments additionally require `python -m pip install -e ".[quantization]"`; the reported 4B results use native bfloat16.

Download the two frozen models:

```bash
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir models/Qwen3-VL-4B-Instruct
hf download google/siglip2-base-patch16-224 --local-dir models/siglip2-base-patch16-224
```

Model pages: [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) and [SigLIP2 base patch16-224](https://huggingface.co/google/siglip2-base-patch16-224).

## Data download and preparation

Datasets are not redistributed in this repository. Follow the original licenses and terms.

### Waterbirds

Waterbirds is released by the [Group DRO project](https://github.com/kohpangwei/group_DRO#waterbirds). Download the prepared `waterbird_complete95_forest2water2` archive from the project link:

```bash
mkdir -p data/waterbirds
curl -L https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz -o data/waterbirds/waterbirds.tar.gz
tar -xzf data/waterbirds/waterbirds.tar.gz -C data/waterbirds
```

The Group DRO README also documents access through [WILDS](https://wilds.stanford.edu/) if the direct archive is unavailable. The extracted folder must contain `metadata.csv` and the class image folders. Convert it to the common manifest:

```bash
python scripts/prepare_waterbirds.py \
  --source data/waterbirds/waterbird_complete95_forest2water2 \
  --output data/waterbirds/manifest.jsonl
```

### Spawrious O2O Easy

Use the [official Spawrious repository](https://github.com/aengusl/spawrious), which provides both direct challenge downloads and a Python downloader:

```bash
python -m pip install "git+https://github.com/aengusl/spawrious.git"
python -c "from spawrious.torch import get_spawrious_dataset; get_spawrious_dataset(dataset_name='o2o_easy', root_dir='data/spawrious_raw')"
```

Point `--source` at the directory containing `spawrious224/0`, `spawrious224/1`, and the OOD scene folders:

```bash
python scripts/prepare_spurious_tasks.py spawrious \
  --source data/spawrious_raw \
  --output data/spawrious_o2o_easy \
  --challenge o2o_easy \
  --full-evaluation \
  --validation-per-group-per-environment 50
```

Run the integrity check before a long evaluation:

```bash
python scripts/verify_manifest_images.py \
  --manifest data/spawrious_o2o_easy/manifest.jsonl \
  --report data/spawrious_o2o_easy/image_verification.json \
  --full-load
```

## Running diagnostic-aware selection

The generic runner caches SigLIP embeddings, evaluates validation, freezes `lambda`, and then evaluates test:

```bash
python scripts/run_multiclass_weighted.py \
  --config configs/spawrious_o2o_easy_qwen3vl4b_bf16.yaml \
  --output artifacts/spawrious_weighted \
  --embedding-cache artifacts/spawrious_weighted/siglip2_embeddings.npz \
  --stage all
```

Replace the config with `configs/waterbirds_qwen3vl4b_bf16.yaml` for Waterbirds. For strict held-out reporting, run stages separately in the order `prepare`, `validation`, `freeze`, and `test`, and never modify the frozen protocol after test begins.

## Running metadata-free failure-contrastive discovery

The method consumes frozen RICES validation predictions and the already cached SigLIP embeddings. Example for Spawrious:

```bash
python scripts/run_failure_contrastive_discovery.py prepare \
  --manifest data/spawrious_o2o_easy/manifest.jsonl \
  --predictions artifacts/spawrious_weighted/predictions/validation/rices_global_3shot.jsonl \
  --embedding-cache artifacts/spawrious_weighted/siglip2_embeddings.npz \
  --output artifacts/spawrious_failure/panels.jsonl

python scripts/run_failure_contrastive_discovery.py induce \
  --config configs/spawrious_o2o_easy_qwen3vl4b_bf16.yaml \
  --panels artifacts/spawrious_failure/panels.jsonl \
  --output artifacts/spawrious_failure/rules.jsonl

python scripts/run_failure_contrastive_discovery.py evaluate \
  --config configs/spawrious_o2o_easy_qwen3vl4b_bf16.yaml \
  --manifest data/spawrious_o2o_easy/manifest.jsonl \
  --predictions artifacts/spawrious_weighted/predictions/validation/rices_global_3shot.jsonl \
  --embedding-cache artifacts/spawrious_weighted/siglip2_embeddings.npz \
  --rules artifacts/spawrious_failure/rules.jsonl \
  --output artifacts/spawrious_failure/validation_predictions.jsonl \
  --summary artifacts/spawrious_failure/validation_summary.json \
  --scope full_routed \
  --punish-beta 2
```

Only confusion pairs that pass the validation rule may be enabled for frozen test inference.

After freezing the validation summary, test evaluation uses the test RICES cache and explicitly supplies that summary:

```bash
python scripts/run_failure_contrastive_discovery.py evaluate \
  --config configs/spawrious_o2o_easy_qwen3vl4b_bf16.yaml \
  --manifest data/spawrious_o2o_easy/manifest.jsonl \
  --predictions artifacts/spawrious_weighted/predictions/test/rices_global_3shot.jsonl \
  --rules artifacts/spawrious_failure/rules.jsonl \
  --validation-summary artifacts/spawrious_failure/validation_summary.json \
  --output artifacts/spawrious_failure/test_predictions.jsonl \
  --summary artifacts/spawrious_failure/test_summary.json \
  --scope frozen_test \
  --punish-beta 2
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover global top-k selection, diagnostic coverage, weighted ranking, deterministic folds, confusion mining, rescue/harm accounting, all-fold gating, and protection against test/group metadata entering selection or routing.

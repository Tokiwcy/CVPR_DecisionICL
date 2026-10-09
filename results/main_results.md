# Audited paper tables (accuracy, %)

Generated from the source-hash-verified experiment ledger. Each cell below contains 1-shot / 2-shot. P→D means parent→DecisionICL, using the same model, split, and shot. CIRCLES is 2-shot only. Values are rounded to 2 decimals for display; calculations use full CSV precision.

| Model | Dataset | Random 1 / 2 | RICES P→D 1 / 2 | MUIER P→D 1 / 2 | CIRCLES P→D 2 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B BF16 | DTD | 48.94 / 50.27 | 76.06→77.55 / 77.87→78.78 | 76.28→77.55 / 78.40→79.04 | 77.18→78.99 |
| Qwen3-VL-4B BF16 | AIRCRAFT | 21.90 / 22.92 | 56.62→65.38 / 59.20→65.71 | 61.45→64.75 / 62.77→65.26 | 57.10→65.56 |
| Qwen3-VL-4B BF16 | CUB | 24.30 / 25.11 | 74.30→75.39 / 79.10→80.77 | 71.01→71.90 / 75.51→76.43 | 75.62→77.05 |
| Qwen3-VL-4B BF16 | DOGS | 46.61 / 46.79 | 72.74→74.37 / 77.41→78.24 | 76.48→76.77 / 81.42→81.56 | 74.15→74.94 |
| Qwen3-VL-4B BF16 | FLOWERS102 | 51.91 / 52.55 | 92.75→93.15 / 97.25→97.28 | 90.96→92.75 / 94.16→96.67 | 94.55→96.05 |
| Qwen2.5-VL-7B NF4 | DTD | 35.00 / 33.35 | 65.05→66.65 / 66.86→70.43 | 66.54→67.82 / 69.41→71.17 | 70.43→72.98 |
| Qwen2.5-VL-7B NF4 | AIRCRAFT | 31.98 / 30.57 | 51.37→53.59 / 57.16→62.44 | 53.62→54.61 / 58.90→61.96 | 50.11→61.00 |
| Qwen2.5-VL-7B NF4 | CUB | 24.34 / 24.72 | 70.34→72.55 / 75.64→78.99 | 67.06→69.36 / 72.32→75.53 | 66.60→73.74 |
| Qwen2.5-VL-7B NF4 | DOGS | 36.27 / 37.31 | 68.66→72.53 / 72.49→75.41 | 74.50→75.08 / 77.94→78.93 | 64.94→70.96 |
| Qwen2.5-VL-7B NF4 | FLOWERS102 | 49.44 / 50.40 | 94.41→94.91 / 97.11→97.76 | 91.87→93.74 / 94.24→97.35 | 87.80→93.58 |
| Gemma4 12B NF4 | DTD | 48.83 / 50.16 | 72.18→74.04 / 73.09→74.57 | 73.83→75.59 / 75.11→76.33 | 71.97→75.27 |
| Gemma4 12B NF4 | AIRCRAFT | 10.26 / 10.50 | 51.49→56.71 / 52.03→59.08 | 57.82→59.71 / 59.83→61.78 | 35.01→46.14 |
| Gemma4 12B NF4 | CUB | 12.58 / 13.16 | 73.61→75.84 / 74.43→77.47 | 69.40→71.84 / 70.61→73.90 | 64.98→72.92 |
| Gemma4 12B NF4 | DOGS | 24.38 / 24.84 | 62.31→66.97 / 66.83→71.64 | 69.98→70.61 / 74.86→75.80 | 53.08→62.05 |
| Gemma4 12B NF4 | FLOWERS102 | 42.45 / 44.25 | 90.89→91.53 / 94.88→95.56 | 89.35→91.58 / 92.34→95.09 | 86.18→92.05 |

## Dataset-level RICES, 2-shot (parent→DecisionICL; gain in pp)

This canonical parent slice shows heterogeneity across datasets; the complete parent-by-shot matrix remains above and in the supplement.

| Dataset | Qwen3-VL-4B BF16 | Qwen2.5-VL-7B NF4 | Gemma4 12B NF4 |
| --- | ---: | ---: | ---: |
| DTD | 77.87→78.78 (+0.90) | 66.86→70.43 (+3.56) | 73.09→74.57 (+1.49) |
| AIRCRAFT | 59.20→65.71 (+6.51) | 57.16→62.44 (+5.28) | 52.03→59.08 (+7.05) |
| CUB | 79.10→80.77 (+1.67) | 75.64→78.99 (+3.36) | 74.43→77.47 (+3.05) |
| DOGS | 77.41→78.24 (+0.83) | 72.49→75.41 (+2.91) | 66.83→71.64 (+4.81) |
| FLOWERS102 | 97.25→97.28 (+0.03) | 97.11→97.76 (+0.65) | 94.88→95.56 (+0.68) |

## Main-paper compact table: mean parent → DecisionICL accuracy

These are equal-dataset descriptive means over five datasets; they are not a pooled confidence interval or a claim that the 75 comparisons are independent.

| Model | RICES 1 / 2 | MUIER 1 / 2 | CIRCLES 2 | Mean gain, all 25 |
| --- | --- | --- | --- | --- |
| Qwen3-VL-4B BF16 | 74.49→77.17 / 78.17→80.15 | 75.23→76.74 / 78.45→79.79 | 75.72→78.52 | +2.06 pp |
| Qwen2.5-VL-7B NF4 | 69.96→72.04 / 73.85→77.00 | 70.72→72.12 / 74.56→76.99 | 67.98→74.45 | +3.11 pp |
| Gemma4 12B NF4 | 70.10→73.02 / 72.25→75.67 | 72.07→73.86 / 74.55→76.58 | 62.24→69.69 | +3.52 pp |

## Generic direct-sample alternative (Qwen3)

Across 25 matched cells, Generic is below DecisionICL in 24/25; its equal-cell mean Generic−DecisionICL is -1.36 pp. This alternative also changes the retrieval decision rule and validation objective, so the comparison does not isolate factor vocabulary alone.

## Fixed one-step class-prototype rotation (Qwen3, 2-shot RICES)

| Dataset | Parent | DecisionICL | Rotated | Rotated−DecisionICL |
| --- | ---: | ---: | ---: | ---: |
| DTD | 77.87 | 78.78 | 76.70 | -2.07 |
| AIRCRAFT | 59.20 | 65.71 | 56.62 | -9.09 |
| CUB | 79.10 | 80.77 | 79.50 | -1.27 |

CUB's validation calibration selected the parent-only endpoint (α=1); its result cannot be attributed solely to incorrect class–factor alignment.


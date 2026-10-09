"""Verify the audited ledger inputs and build paper-facing descriptive tables."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LEDGER = ROOT / "docs" / "experiment_results_ledger_20260920_zh"
MODELS = ("Qwen3-VL-4B BF16", "Qwen2.5-VL-7B NF4", "Gemma4 12B NF4")
DATASETS = ("dtd", "aircraft", "cub", "dogs", "flowers102")
PARENTS = ("RICES", "MUIER", "CIRCLES")

METHODS = {
    "Random": ("random_official_prompt", "random"),
    "RICES": ("rices_clip_official_prompt", "rices"),
    "RICES+Decision": ("rices_clip_atom_class_equal_raw_official_prompt", "rices+matched_decision"),
    "MUIER": ("muier_classification_adapted_siglip_official_prompt", "muier"),
    "MUIER+Decision": (
        "muier_classification_adapted_siglip_atom_class_equal_raw_official_prompt",
        "muier+matched_decision",
    ),
    "CIRCLES": ("circles_official_prompt", "circles"),
    "CIRCLES+Decision": ("circles_atom_class_equal_raw_official_prompt", "circles+matched_decision"),
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_sources() -> dict:
    manifest = json.loads(LEDGER.with_suffix(".sources.json").read_text(encoding="utf-8-sig"))
    expected = {
        "main_qwen_method_results": 120,
        "gemma_independently_audited_method_results": 60,
        "generic_control_method_results": 25,
        "rotated_factor_class_audited_method_results": 3,
        "visual_centroid_class_audited_method_results": 3,
    }
    for key, value in expected.items():
        assert manifest[key] == value, (key, manifest[key], value)
    gate_count = manifest.get("parent_class_gate_audited_method_results", 0)
    assert 0 <= gate_count <= 3
    assert manifest["structured_result_rows"] == 211 + gate_count
    assert manifest["paired_result_rows"] == 143 + 2 * gate_count
    pending = manifest["pending_ablation_stages"]
    assert pending == ([] if gate_count == 3 else ["qwen3_parent_class_gate_2shot"])
    for relative, digest in manifest["sources_sha256"].items():
        path = ROOT / relative
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
    return manifest


def normalized_method(raw: str) -> str | None:
    for label, names in METHODS.items():
        if raw in names:
            return label
    return None


def main_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, int, str], float]:
    output = {}
    for row in rows:
        method = normalized_method(row["method"])
        if method is None:
            continue
        key = (row["model"], row["dataset"], int(row["shots"]), method)
        assert key not in output, key
        output[key] = float(row["accuracy"]) * 100
    for model in MODELS:
        for dataset in DATASETS:
            for shot in (1, 2):
                names = ("Random", "RICES", "RICES+Decision", "MUIER", "MUIER+Decision")
                if shot == 2:
                    names += ("CIRCLES", "CIRCLES+Decision")
                for name in names:
                    assert (model, dataset, shot, name) in output, (model, dataset, shot, name)
    assert len(output) == 180, len(output)
    return output


def value(main: dict, model: str, dataset: str, shot: int, name: str) -> float:
    return main[(model, dataset, shot, name)]


def build(main: dict, rows: list[dict[str, str]], paired: list[dict[str, str]]) -> tuple[str, dict]:
    lines = [
        "# Audited paper tables (accuracy, %)",
        "",
        "Generated from the source-hash-verified experiment ledger. Each cell below contains 1-shot / 2-shot. P→D means parent→Decision, using the same model, split, and shot. CIRCLES is 2-shot only. Values are rounded to 2 decimals for display; calculations use full CSV precision.",
        "",
        "| Model | Dataset | Random 1 / 2 | RICES P→D 1 / 2 | MUIER P→D 1 / 2 | CIRCLES P→D 2 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    means = defaultdict(list)
    mean_pairs = defaultdict(list)
    gains = []
    for model in MODELS:
        for dataset in DATASETS:
            def pair(shot: int, parent: str) -> str:
                p = value(main, model, dataset, shot, parent)
                d = value(main, model, dataset, shot, parent + "+Decision")
                gains.append(d - p)
                means[(model, shot, parent)].append(d - p)
                mean_pairs[(model, shot, parent)].append((p, d))
                return f"{p:.2f}→{d:.2f}"

            random_cell = " / ".join(f"{value(main, model, dataset, shot, 'Random'):.2f}" for shot in (1, 2))
            rices_cell = " / ".join(pair(shot, "RICES") for shot in (1, 2))
            muier_cell = " / ".join(pair(shot, "MUIER") for shot in (1, 2))
            circles_cell = pair(2, "CIRCLES")
            lines.append(f"| {model} | {dataset.upper()} | {random_cell} | {rices_cell} | {muier_cell} | {circles_cell} |")
    assert len(gains) == 75 and all(g > 0 for g in gains)
    lines += ["", "## Dataset-level RICES, 2-shot (parent→Decision; gain in pp)", "", "This canonical parent slice shows heterogeneity across datasets; the complete parent-by-shot matrix remains above and in the supplement.", "", "| Dataset | Qwen3-VL-4B BF16 | Qwen2.5-VL-7B NF4 | Gemma4 12B NF4 |", "| --- | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        cells = []
        for model in MODELS:
            p = value(main, model, dataset, 2, "RICES")
            d = value(main, model, dataset, 2, "RICES+Decision")
            cells.append(f"{p:.2f}→{d:.2f} ({d-p:+.2f})")
        lines.append(f"| {dataset.upper()} | " + " | ".join(cells) + " |")
    lines += ["", "## Main-paper compact table: mean parent → Decision accuracy", "", "These are equal-dataset descriptive means over five datasets; they are not a pooled confidence interval or a claim that the 75 comparisons are independent.", "", "| Model | RICES 1 / 2 | MUIER 1 / 2 | CIRCLES 2 | Mean gain, all 25 |", "| --- | --- | --- | --- | --- |"]
    model_means = {}
    for model in MODELS:
        model_gains = [x for (m, _, _), vals in means.items() if m == model for x in vals]
        assert len(model_gains) == 25
        model_means[model] = sum(model_gains) / 25
        def mean_pair(shot: int, parent: str) -> str:
            vals = mean_pairs[(model, shot, parent)]
            assert len(vals) == 5
            return f"{sum(p for p, _ in vals)/5:.2f}→{sum(d for _, d in vals)/5:.2f}"
        lines.append(
            f"| {model} | {mean_pair(1,'RICES')} / {mean_pair(2,'RICES')} | "
            f"{mean_pair(1,'MUIER')} / {mean_pair(2,'MUIER')} | "
            f"{mean_pair(2,'CIRCLES')} | +{model_means[model]:.2f} pp |"
        )

    qwen = MODELS[0]
    generic_names = {
        "RICES": "rices+generic_visual_direct_sample",
        "MUIER": "muier+generic_visual_direct_sample",
        "CIRCLES": "circles+generic_visual_direct_sample",
    }
    qwen_rows = {(r["dataset"], int(r["shots"]), r["method"]): float(r["accuracy"]) * 100 for r in rows if r["model"] == qwen}
    generic_deltas = []
    generic_lower = 0
    for dataset in DATASETS:
        for shot in (1, 2):
            for parent in (PARENTS if shot == 2 else PARENTS[:2]):
                generic = qwen_rows[(dataset, shot, generic_names[parent])]
                decision = value(main, qwen, dataset, shot, parent + "+Decision")
                delta = generic - decision
                generic_deltas.append(delta)
                generic_lower += delta < 0
    assert len(generic_deltas) == 25 and generic_lower == 24
    lines += ["", "## Generic direct-sample alternative (Qwen3)", "", f"Across 25 matched cells, Generic is below Decision in {generic_lower}/25; its equal-cell mean Generic−Decision is {sum(generic_deltas)/25:+.2f} pp. This alternative also changes the retrieval decision rule and validation objective, so the comparison does not isolate factor vocabulary alone.", ""]

    lines += ["## Fixed one-step class-prototype rotation (Qwen3, 2-shot RICES)", "", "| Dataset | Parent | Decision | Rotated | Rotated−Decision |", "| --- | ---: | ---: | ---: | ---: |"]
    rotation = {}
    for dataset in DATASETS[:3]:
        rotated = qwen_rows[(dataset, 2, "rices+rotated_decision_class_prototype")]
        parent = value(main, qwen, dataset, 2, "RICES")
        decision = value(main, qwen, dataset, 2, "RICES+Decision")
        rotation[dataset] = rotated - decision
        lines.append(f"| {dataset.upper()} | {parent:.2f} | {decision:.2f} | {rotated:.2f} | {rotated-decision:+.2f} |")
    lines += ["", "CUB's validation calibration selected the parent-only endpoint (α=1); its result cannot be attributed solely to incorrect class–factor alignment.", ""]

    decision_pairs = [r for r in paired if r["method"].lower().endswith("+decision")]
    assert len(decision_pairs) == 75, len(decision_pairs)
    assert all(float(r["delta_pp"]) > 0 for r in decision_pairs)
    raw_significant = {}
    for model in MODELS:
        relevant = [r for r in decision_pairs if r["model"] == model]
        assert len(relevant) == 25
        raw_significant[model] = sum(float(r["exact_mcnemar_p"]) < 0.05 for r in relevant)
    numbers = {
        "ledger_source_count": len(json.loads(LEDGER.with_suffix(".sources.json").read_text(encoding="utf-8-sig"))["sources_sha256"]),
        "main_result_count": len(main),
        "decision_parent_pairs": len(gains),
        "positive_decision_parent_pairs": sum(g > 0 for g in gains),
        "model_mean_gain_pp": model_means,
        "model_raw_p_below_0_05_count": raw_significant,
        "generic_compared_cells": len(generic_deltas),
        "generic_below_decision_cells": generic_lower,
        "generic_minus_decision_mean_pp": sum(generic_deltas) / len(generic_deltas),
        "rotation_minus_decision_pp": rotation,
    }
    return "\n".join(lines) + "\n", numbers


def supplementary(rows: list[dict[str, str]], paired: list[dict[str, str]]) -> str:
    lines = [
        "# Supplementary audited result tables",
        "",
        "Automatically generated from the audited ledger CSV after source-hash verification. Accuracy and macro-accuracy are percentages; invalid is the number of non-exact class outputs. Paired differences and confidence intervals are percentage points. McNemar values are unadjusted two-sided exact p-values and should be interpreted per comparison.",
        "",
        "## Every reported method result",
        "",
        "| Model | Dataset | Shot | Method | Accuracy | Macro | Invalid |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['dataset'].upper()} | {row['shots']} | {row['method']} | "
            f"{100*float(row['accuracy']):.3f} | {100*float(row['macro_accuracy']):.3f} | {row['invalid_outputs']} |"
        )
    lines += [
        "", "## Every paired comparison", "",
        "Rescue is correct only for the named method; harm is correct only for its comparator.", "",
        "| Model | Dataset | Shot | Method vs comparator | Δ accuracy | Rescue / harm | Bootstrap 95% CI | Exact p |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for row in paired:
        lines.append(
            f"| {row['model']} | {row['dataset'].upper()} | {row['shots']} | "
            f"{row['method']} vs {row['comparator']} | {float(row['delta_pp']):+.3f} | "
            f"{row['rescue']} / {row['harm']} | [{float(row['ci_low_pp']):+.3f}, {float(row['ci_high_pp']):+.3f}] | "
            f"{float(row['exact_mcnemar_p']):.4g} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    verify_sources()
    rows = load_csv(LEDGER.with_suffix(".csv"))
    paired = load_csv(Path(str(LEDGER) + ".paired.csv"))
    manifest = json.loads(LEDGER.with_suffix(".sources.json").read_text(encoding="utf-8-sig"))
    gate_count = manifest.get("parent_class_gate_audited_method_results", 0)
    assert len(rows) == 211 + gate_count and len(paired) == 143 + 2 * gate_count
    table, numbers = build(main_rows(rows), rows, paired)
    (HERE / "tables.md").write_text(table, encoding="utf-8")
    (HERE / "supplementary_tables.md").write_text(supplementary(rows, paired), encoding="utf-8")
    (HERE / "paper_numbers.json").write_text(json.dumps(numbers, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(numbers, ensure_ascii=True))


if __name__ == "__main__":
    main()

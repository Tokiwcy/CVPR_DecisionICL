"""Export hash-verified DecisionICL paper results from a BehaviorICL workspace.

The repository contains generated tables, original review inputs and portable
verifiers; full raw/cache/numeric evidence is supplied through release assets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


MAIN_METHODS = {
    "random_official_prompt": "Random",
    "random": "Random",
    "rices_clip_official_prompt": "RICES",
    "rices": "RICES",
    "muier_classification_adapted_siglip_official_prompt": "MUIER",
    "muier": "MUIER",
    "rices_clip_atom_class_equal_raw_official_prompt": "RICES+DecisionICL",
    "rices+matched_decision": "RICES+DecisionICL",
    "muier_classification_adapted_siglip_atom_class_equal_raw_official_prompt": "MUIER+DecisionICL",
    "muier+matched_decision": "MUIER+DecisionICL",
    "circles_official_prompt": "CIRCLES",
    "circles": "CIRCLES",
    "circles_atom_class_equal_raw_official_prompt": "CIRCLES+DecisionICL",
    "circles+matched_decision": "CIRCLES+DecisionICL",
    "rices+generic_visual_direct_sample": "RICES+Generic",
    "muier+generic_visual_direct_sample": "MUIER+Generic",
    "circles+generic_visual_direct_sample": "CIRCLES+Generic",
    "rices+rotated_decision_class_prototype": "RICES+RotatedFactorClass",
    "rices+clip_centroid_class_gate": "RICES+CLIPCentroidGate",
    "rices+parent_top1_class_gate_no_factors": "RICES+NoFactorClassGate",
}

PAIRED_NAMES = {
    "RICES+RotatedDecision": "RICES+RotatedFactorClass",
    "RICES+CLIPCentroidGate": "RICES+CLIPCentroidGate",
    "RICES+NoFactorClassGate": "RICES+NoFactorClassGate",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def public_pair_name(name: str) -> str:
    if name in PAIRED_NAMES:
        return PAIRED_NAMES[name]
    standard = name.replace("+Decision", "+DecisionICL")
    if standard.startswith("rices"):
        standard = "RICES" + standard[5:]
    elif standard.startswith("muier"):
        standard = "MUIER" + standard[5:]
    elif standard.startswith("circles"):
        standard = "CIRCLES" + standard[7:]
    return standard


def verify_workspace(root: Path) -> tuple[Path, dict]:
    ledger = root / "docs/experiment_results_ledger_20260920_zh"
    manifest = json.loads(ledger.with_suffix(".sources.json").read_text(encoding="utf-8"))
    expected = {
        "structured_result_rows": 214,
        "paired_result_rows": 149,
        "main_qwen_method_results": 120,
        "gemma_independently_audited_method_results": 60,
        "generic_control_method_results": 25,
        "rotated_factor_class_audited_method_results": 3,
        "visual_centroid_class_audited_method_results": 3,
        "parent_class_gate_audited_method_results": 3,
    }
    for key, count in expected.items():
        if manifest.get(key) != count:
            raise ValueError(f"source ledger {key}: {manifest.get(key)} != {count}")
    for relative, digest in manifest["sources_sha256"].items():
        source = root / relative
        if not source.is_file() or sha(source) != digest:
            raise ValueError("audited source SHA mismatch: " + relative)
    return ledger, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    destination = Path(__file__).resolve().parents[1] / "results"
    destination.mkdir(parents=True, exist_ok=True)
    ledger, manifest = verify_workspace(workspace)
    original_methods = read_csv(ledger.with_suffix(".csv"))
    original_pairs = read_csv(Path(str(ledger) + ".paired.csv"))
    if len(original_methods) != 214 or len(original_pairs) != 149:
        raise ValueError("ledger row count drift")

    methods = []
    for row in original_methods:
        source_method = row["method"]
        if source_method not in MAIN_METHODS:
            raise ValueError("unmapped source method: " + source_method)
        methods.append({
            "model": row["model"], "dataset": row["dataset"], "shots": row["shots"],
            "method": MAIN_METHODS[source_method], "source_method": source_method,
            "accuracy": row["accuracy"], "macro_accuracy": row["macro_accuracy"],
            "invalid_outputs": row["invalid_outputs"],
            "source_artifact": row["source"],
        })
    pairs = []
    for row in original_pairs:
        pairs.append({
            "model": row["model"], "dataset": row["dataset"], "shots": row["shots"],
            "method": public_pair_name(row["method"]),
            "comparator": public_pair_name(row["comparator"]),
            "source_method": row["method"], "source_comparator": row["comparator"],
            "delta_pp": row["delta_pp"], "rescue": row["rescue"], "harm": row["harm"],
            "ci_low_pp": row["ci_low_pp"], "ci_high_pp": row["ci_high_pp"],
            "exact_mcnemar_p": row["exact_mcnemar_p"],
            "source_artifact": row["source"],
        })
    method_path = destination / "method_results.csv"
    pair_path = destination / "paired_results.csv"
    write_csv(method_path, list(methods[0]), methods)
    write_csv(pair_path, list(pairs[0]), pairs)

    tables = (workspace / "paper/cvpr2027/tables.md").read_text(encoding="utf-8")
    numbers = json.loads((workspace / "paper/cvpr2027/paper_numbers.json").read_text(encoding="utf-8"))
    if numbers["ledger_source_count"] != len(manifest["sources_sha256"]):
        raise ValueError("paper tables were not regenerated from the current source ledger")
    if numbers["main_result_count"] != 180 or numbers["decision_parent_pairs"] != 75:
        raise ValueError("paper main result matrix drift")
    tables = tables.replace("Decision", "DecisionICL")
    (destination / "main_results.md").write_text(tables, encoding="utf-8")

    supplemental = [
        "# DecisionICL supplementary results", "",
        "All 214 method rows and 149 paired comparisons from the audited ledger. Accuracy, macro accuracy, Δ, and CI are percentages/percentage points as indicated. Exact McNemar p values are two-sided and unadjusted.", "",
        "## Method results", "",
        "| Model | Dataset | Shot | Method | Accuracy (%) | Macro (%) | Invalid |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in methods:
        supplemental.append(
            f"| {row['model']} | {row['dataset'].upper()} | {row['shots']} | {row['method']} | "
            f"{100*float(row['accuracy']):.3f} | {100*float(row['macro_accuracy']):.3f} | {row['invalid_outputs']} |"
        )
    supplemental += [
        "", "## Paired results", "",
        "Rescue means the named method corrects a comparator error; harm means it changes a comparator-correct answer to an error.", "",
        "| Model | Dataset | Shot | Method vs comparator | Δ (pp) | Rescue / harm | Bootstrap 95% CI (pp) | Exact p |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for row in pairs:
        supplemental.append(
            f"| {row['model']} | {row['dataset'].upper()} | {row['shots']} | "
            f"{row['method']} vs {row['comparator']} | {float(row['delta_pp']):+.3f} | "
            f"{row['rescue']} / {row['harm']} | "
            f"[{float(row['ci_low_pp']):+.3f}, {float(row['ci_high_pp']):+.3f}] | "
            f"{float(row['exact_mcnemar_p']):.4g} |"
        )
    (destination / "supplementary_results.md").write_text("\n".join(supplemental) + "\n", encoding="utf-8")
    (destination / "paper_numbers.json").write_text(
        json.dumps(numbers, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    matched_path = workspace / "paper/cvpr2027/matched_source_control.json"
    matched = json.loads(matched_path.read_text(encoding="utf-8"))
    if len(matched) != 4 or {row["dataset"] for row in matched} != {"dtd", "aircraft"}:
        raise ValueError("matched factor-source control drift")
    matched = [
        {key.replace("decision_", "decisionicl_"): value for key, value in row.items()}
        for row in matched
    ]
    (destination / "matched_source_control.json").write_text(
        json.dumps(matched, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    input_files = {
        "ledger_method_csv": sha(ledger.with_suffix(".csv")),
        "ledger_paired_csv": sha(Path(str(ledger) + ".paired.csv")),
        "ledger_source_manifest": sha(ledger.with_suffix(".sources.json")),
        "paper_tables": sha(workspace / "paper/cvpr2027/tables.md"),
        "paper_numbers": sha(workspace / "paper/cvpr2027/paper_numbers.json"),
        "matched_source_control": sha(matched_path),
    }
    output_files = {name: sha(destination / name) for name in (
        "method_results.csv", "paired_results.csv", "main_results.md",
        "supplementary_results.md", "paper_numbers.json", "source_manifest.json",
        "matched_source_control.json",
    )}
    export = {
        "method_name": "DecisionICL",
        "source_ledger_generated_at": manifest["generated_at"],
        "source_files_verified": len(manifest["sources_sha256"]),
        "main_method_results": 180,
        "main_decisionicl_parent_pairs": 75,
        "control_method_results": 34,
        "paired_result_rows": 149,
        "qwen3_three_seed_additional_results_audited_but_not_aggregated": manifest.get(
            "qwen3_seed_repeat_independently_audited_method_results", 0
        ),
        "source_sha256": input_files,
        "export_sha256": output_files,
    }
    (destination / "export_manifest.json").write_text(
        json.dumps(export, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: export[key] for key in (
        "source_files_verified", "main_method_results", "control_method_results",
        "paired_result_rows", "qwen3_three_seed_additional_results_audited_but_not_aggregated"
    )}, ensure_ascii=True))


if __name__ == "__main__":
    main()

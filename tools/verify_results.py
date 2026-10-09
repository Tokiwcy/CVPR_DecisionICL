"""Check the portable DecisionICL publication bundle without private artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ("Qwen3-VL-4B BF16", "Qwen2.5-VL-7B NF4", "Gemma4 12B NF4")
DATASETS = ("dtd", "aircraft", "cub", "dogs", "flowers102")
ONE = {"Random", "RICES", "MUIER", "RICES+DecisionICL", "MUIER+DecisionICL"}
TWO = ONE | {"CIRCLES", "CIRCLES+DecisionICL"}
CONTROLS = {
    "RICES+Generic": 10,
    "MUIER+Generic": 10,
    "CIRCLES+Generic": 5,
    "RICES+RotatedFactorClass": 3,
    "RICES+CLIPCentroidGate": 3,
    "RICES+NoFactorClassGate": 3,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    exported = json.loads((RESULTS / "export_manifest.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((RESULTS / "source_manifest.json").read_text(encoding="utf-8"))
    numbers = json.loads((RESULTS / "paper_numbers.json").read_text(encoding="utf-8"))
    matched = json.loads((RESULTS / "matched_source_control.json").read_text(encoding="utf-8"))
    assert exported["method_name"] == "DecisionICL"
    assert exported["source_files_verified"] == numbers["ledger_source_count"] == 152
    assert len(source_manifest["sources_sha256"]) == 152, "source-hash closure count changed"
    assert len(matched) == 4 and {row["dataset"] for row in matched} == {"dtd", "aircraft"}
    assert all(row["complete_sha256"] and row["independent_audit_sha256"] for row in matched)
    for name, expected in exported["export_sha256"].items():
        assert digest(RESULTS / name) == expected, f"export hash mismatch: {name}"

    code_root = ROOT / "research_code" / "frozen"
    code_manifest = json.loads((code_root / "snapshot_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in code_manifest["files"].items():
        assert digest(code_root / relative) == expected, f"code hash mismatch: {relative}"

    with (RESULTS / "method_results.csv").open(newline="", encoding="utf-8") as stream:
        methods = list(csv.DictReader(stream))
    with (RESULTS / "paired_results.csv").open(newline="", encoding="utf-8") as stream:
        paired = list(csv.DictReader(stream))
    assert len(methods) == 214 and len(paired) == 149
    assert exported["main_method_results"] == numbers["main_result_count"] == 180
    assert exported["paired_result_rows"] == len(paired)
    counts = Counter(row["method"] for row in methods)
    for name, expected in CONTROLS.items():
        assert counts[name] == expected, (name, counts[name])

    index: dict[tuple[str, str, int, str], dict[str, str]] = {}
    for row in methods:
        key = (row["model"], row["dataset"], int(row["shots"]), row["method"])
        assert key not in index, f"duplicate method row: {key}"
        assert row["source_method"] and row["source_artifact"]
        assert 0 <= float(row["accuracy"]) <= 1
        assert 0 <= float(row["macro_accuracy"]) <= 1
        assert int(row["invalid_outputs"]) >= 0
        index[key] = row

    main_pairs = []
    for model in MODELS:
        for dataset in DATASETS:
            for shots, expected in ((1, ONE), (2, TWO)):
                actual = {name for m, d, s, name in index if (m, d, s) == (model, dataset, shots)}
                assert actual == expected | ({"RICES+Generic", "MUIER+Generic"} if model == MODELS[0] and shots == 1 else set()) | ({"RICES+Generic", "MUIER+Generic", "CIRCLES+Generic"} if model == MODELS[0] and shots == 2 else set()) | ({"RICES+RotatedFactorClass", "RICES+CLIPCentroidGate", "RICES+NoFactorClassGate"} if model == MODELS[0] and dataset in DATASETS[:3] and shots == 2 else set()), (model, dataset, shots, actual)

    for row in paired:
        method = row["method"]
        comparator = row["comparator"]
        key = (row["model"], row["dataset"], int(row["shots"]))
        assert method and comparator and row["source_method"] and row["source_comparator"]
        assert int(row["rescue"]) >= 0 and int(row["harm"]) >= 0
        assert 0 <= float(row["exact_mcnemar_p"]) <= 1
        assert float(row["ci_low_pp"]) <= float(row["ci_high_pp"])
        if method.endswith("+DecisionICL") and comparator in {"RICES", "MUIER", "CIRCLES"}:
            a = float(index[(*key, method)]["accuracy"])
            b = float(index[(*key, comparator)]["accuracy"])
            delta = float(row["delta_pp"])
            assert abs(delta - 100 * (a - b)) < 1e-6, (key, method)
            assert delta > 0, (key, method)
            main_pairs.append(row)

    assert len(main_pairs) == numbers["decision_parent_pairs"] == 75
    assert len(main_pairs) == numbers["positive_decision_parent_pairs"]
    for model in MODELS:
        values = [float(row["delta_pp"]) for row in main_pairs if row["model"] == model]
        assert len(values) == 25
        assert abs(sum(values) / len(values) - numbers["model_mean_gain_pp"][model]) < 1e-6

    print(json.dumps({
        "status": "passed", "source_hashes_recorded": len(source_manifest["sources_sha256"]),
        "frozen_code_files_verified": len(code_manifest["files"]),
        "method_results": len(methods), "paired_results": len(paired),
        "main_pairs": len(main_pairs),
        "paused_three_seed_additional_audited": exported["qwen3_three_seed_additional_results_audited_but_not_aggregated"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

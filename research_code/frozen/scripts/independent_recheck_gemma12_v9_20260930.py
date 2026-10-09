"""Read-only independent recheck of a completed Gemma v9 target block."""
import argparse
import json
import time
from pathlib import Path

import gemma12_cache_provenance_20260926 as provenance
import run_gemma12_main_eval_v7_20260928 as frozen
from dgdr.metrics import accuracy_metrics, paired_comparison

SCRIPT = Path(__file__).resolve()
ROOT = frozen.ROOT.parent / "evaluation_v9_isolated_resume_20260929"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["aircraft", "cub", "dogs", "flowers102"])
    parser.add_argument("shot", type=int, choices=[1, 2])
    args = parser.parse_args()
    frozen.ROOT = ROOT
    _, imports = frozen.setup()
    base = frozen.base
    core = base.core
    ds, shot = args.dataset, args.shot
    out = ROOT / f"{shot}shot" / ds
    complete = out / "complete.json"
    assert core.verified(ROOT / "run.freeze.json") and core.verified(imports)
    assert core.verified(complete), "complete hash closure failed"
    assert core.verified(out / "plan.freeze.json"), "plan hash closure failed"
    audit = read(out / "independent_audit.json")
    plan_audit = read(out / "independent_plan_audit.json")
    results = read(out / "results.json")
    predictions = read(out / "predictions.json")
    assert audit["status"] == plan_audit["status"] == "passed"
    assert plan_audit["test_labels_used"] is False
    assert plan_audit["plan_freeze_sha256"] == core.sha(out / "plan.freeze.json")
    assert audit["complete_sha256"] == core.sha(complete)
    assert audit["results_sha256"] == core.sha(out / "results.json")
    assert audit["predictions_sha256"] == core.sha(out / "predictions.json")
    assert results["test_used_for_tuning"] is False
    assert results["inherited_weights"] is False
    provenance_result = provenance.audit_cache(base, ds, shot, True)
    assert provenance_result == audit["cache_provenance"]
    truth = {x.sample_id: x.label for x in core.manifest(ds, "test")}
    methods = base.METHODS[:5 if shot == 1 else 7]
    assert set(predictions) == set(results["methods"]) == set(audit["methods"]) == set(methods)
    checked = {}
    for method in methods:
        pred = predictions[method]
        assert set(pred) == set(truth)
        metrics = accuracy_metrics(list(truth.values()), [pred[q] for q in truth])
        saved = results["methods"][method]
        for name, value in metrics.items():
            assert abs(value - saved["metrics"][name]) < 1e-12
            assert abs(value - audit["methods"][method]["metrics"][name]) < 1e-12
        invalid = sum(v == "__invalid_exact_output__" for v in pred.values())
        assert invalid == saved["invalid_exact_output_count"] == audit["methods"][method]["invalid"]
        item = {"accuracy": metrics["accuracy"], "macro_accuracy": metrics["macro_accuracy"], "invalid": invalid}
        if "+" in method:
            parent = method.split("+")[0]
            pair = paired_comparison(labels=list(truth.values()),
                                     baseline_predictions=[predictions[parent][q] for q in truth],
                                     method_predictions=[pred[q] for q in truth])
            assert pair == saved["paired_vs_parent"] == audit["methods"][method]["paired_vs_parent"]
            item["paired_vs_parent"] = pair
        checked[method] = item
    receipt = {
        "status": "passed", "dataset": ds, "shots": shot, "time": time.time(),
        "queries": len(truth), "methods": checked,
        "complete_sha256": core.sha(complete), "audit_sha256": core.sha(out / "independent_audit.json"),
        "results_sha256": core.sha(out / "results.json"),
        "predictions_sha256": core.sha(out / "predictions.json"),
        "plan_audit_sha256": core.sha(out / "independent_plan_audit.json"),
        "cache_provenance": provenance_result,
        "source_sha256": core.sha(SCRIPT), "test_used_for_tuning": False,
    }
    path = out / "independent_recheck_20260930.json"
    if path.exists():
        assert read(path)["status"] == "passed"
        raise RuntimeError("recheck receipt exists; do not overwrite")
    core.write(path, receipt)
    print(json.dumps({"status": "passed", "dataset": ds, "shots": shot,
                      "queries": len(truth), "methods": len(checked),
                      "cached_contexts": provenance_result["cached_contexts"],
                      "raw_batches": provenance_result["raw_batches"]}), flush=True)


if __name__ == "__main__":
    main()

"""Hash-checked isolated continuation of the user-paused Gemma v9 run.

The frozen v7 runner still owns the model, prompts, retrieval, calibration,
batching, and target inference. This adapter only checks and copies state.
"""
import argparse
import faulthandler
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import gemma12_cache_provenance_20260926 as provenance
import run_gemma12_main_eval_v7_20260928 as frozen


SCRIPT = Path(__file__).resolve()
SOURCE = frozen.ROOT.parent / "evaluation_v9_isolated_resume_20260929"
ROOT = SOURCE.parent / "evaluation_v10_isolated_resume_20261001"
MANIFEST = ROOT / "resume_source_hashes.json"
RECEIPT = ROOT / "resume_preflight.json"
PAUSE = SOURCE / "PAUSED_BY_USER.json"
PRIOR_SCRIPT = SCRIPT.with_name("run_gemma12_main_eval_v9_resume_20260929.py")
EXCLUDE = {
    "PAUSED_BY_USER.json", "DIAGNOSTIC_HOLD.json", "WATCHDOG_STOP",
    "UNEXPECTED_EXIT_20260929.json", "status.json", "resume.stdout.log",
    "resume.stderr.log", "resume_launch.json", "resume_source_hashes.json",
    "resume_preflight.json",
}
EXPECTED = {"cached_contexts": 4552, "expected_contexts": 46784,
            "raw_batches": 2276, "complete": False}
COMPLETED = [("aircraft", 1), ("aircraft", 2), ("cub", 1),
             ("cub", 2), ("dogs", 1)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n",
                          encoding="utf-8")


def source_files():
    return sorted((p for p in SOURCE.rglob("*")
                   if p.is_file() and p.relative_to(SOURCE).parts[0] not in EXCLUDE),
                  key=lambda p: p.relative_to(SOURCE).as_posix())


def check_bytes(entries):
    assert len(entries) > 50000, "unexpectedly small source state"
    actual = {p.relative_to(SOURCE).as_posix() for p in source_files()}
    assert actual == set(entries), "source file set drift"
    for relative, expected in entries.items():
        src, dst = SOURCE / relative, ROOT / relative
        assert src.is_file() and dst.is_file(), f"missing copied source: {relative}"
        assert sha(src) == expected and sha(dst) == expected, f"copied bytes drift: {relative}"


def bind(root):
    frozen.ROOT = root
    _, imports = frozen.setup()
    base = frozen.base
    assert base.ROOT == root
    assert read(root / "contract.json") == json.loads(json.dumps(base.CONTRACT))
    assert base.core.verified(root / "run.freeze.json")
    assert base.core.verified(imports)
    return base, imports


def completed_blocks(base, root):
    core = base.core
    for shot in (1, 2):
        frozen.source_block("dtd", shot)
    count = 12
    for ds, shot in COMPLETED:
        folder = root / f"{shot}shot" / ds
        complete = folder / "complete.json"
        assert core.verified(complete), f"complete hash closure failed: {folder}"
        assert core.verified(folder / "plan.freeze.json")
        audit = read(folder / "independent_audit.json")
        assert audit["status"] == "passed"
        assert audit["complete_sha256"] == sha(complete)
        assert audit["results_sha256"] == sha(folder / "results.json")
        assert audit["predictions_sha256"] == sha(folder / "predictions.json")
        assert audit["cache_provenance"]["complete"] is True
        plan_audit = read(folder / "independent_plan_audit.json")
        assert plan_audit["status"] == "passed" and plan_audit["test_labels_used"] is False
        if (ds, shot) in (("cub", 2), ("dogs", 1)):
            recheck = read(folder / "independent_recheck_20260930.json")
            assert recheck["status"] == "passed"
            assert recheck["complete_sha256"] == sha(complete)
            assert recheck["audit_sha256"] == sha(folder / "independent_audit.json")
        count += 5 if shot == 1 else 7
    assert count == 41
    partial = root / "2shot" / "dogs"
    assert not (partial / "complete.json").exists()
    assert not (partial / "independent_audit.json").exists()
    return count


def partial_cache(base, root):
    result = provenance.audit_cache(base, "dogs", 2, False)
    assert result == EXPECTED, result
    pause = read(PAUSE)
    assert sha(root / "2shot/dogs/context_cache.jsonl") == pause["context_cache_sha256"]
    assert sha(root / "2shot/dogs/batches/002275.json") == pause["last_raw_batch_sha256"]
    assert sha(root / "2shot/dogs/plan.freeze.json") == pause["plan_freeze_sha256"]
    plan_audit = read(root / "2shot/dogs/independent_plan_audit.json")
    assert plan_audit["status"] == "passed" and plan_audit["test_labels_used"] is False
    return result


def preflight():
    pause = read(PAUSE)
    assert pause["status"] == "paused_by_user" and pause["dataset"] == "dogs"
    assert pause["shots"] == 2 and pause["durable_cached_contexts"] == 4552
    assert pause["raw_batch_files"] == 2276 and pause["audited_method_results"] == 41
    assert pause["test_used_for_tuning"] is False
    prior = read(SOURCE / "resume_preflight.json")
    assert prior["status"] == "passed"
    assert prior["resume_adapter_sha256"] == sha(PRIOR_SCRIPT)
    assert prior["frozen_v7_runner_sha256"] == sha(frozen.__file__)
    base, _ = bind(SOURCE)
    assert completed_blocks(base, SOURCE) == 41
    assert partial_cache(base, SOURCE) == EXPECTED
    if ROOT.exists():
        assert MANIFEST.is_file() and not RECEIPT.exists(), "v10 root already initialized"
        manifest = read(MANIFEST)
        assert manifest["source_root"] == str(SOURCE)
        assert manifest["resume_root"] == str(ROOT)
        entries = manifest["files"]
    else:
        shutil.copytree(SOURCE, ROOT,
                        ignore=lambda directory, names: EXCLUDE & set(names)
                        if Path(directory).resolve() == SOURCE.resolve() else set())
        entries = {p.relative_to(SOURCE).as_posix(): sha(p) for p in source_files()}
        write(MANIFEST, {"source_root": str(SOURCE), "resume_root": str(ROOT),
                         "files": entries})
    check_bytes(entries)
    base, imports = bind(ROOT)
    assert completed_blocks(base, ROOT) == 41
    checked = partial_cache(base, ROOT)
    receipt = {
        "version": "gemma12_v10_isolated_resume_20261001", "status": "passed",
        "checked_at_unix": time.time(), "source_paused_receipt_sha256": sha(PAUSE),
        "source_manifest_sha256": sha(MANIFEST), "source_file_count": len(entries),
        "frozen_v7_runner_sha256": sha(frozen.__file__),
        "resume_adapter_sha256": sha(SCRIPT),
        "contract_sha256": sha(ROOT / "contract.json"),
        "run_freeze_sha256": sha(ROOT / "run.freeze.json"),
        "imported_blocks_freeze_sha256": sha(imports),
        "plan_freeze_sha256": sha(ROOT / "2shot/dogs/plan.freeze.json"),
        "partial_cache_provenance": checked,
        "independently_audited_methods_before_resume": 41,
        "test_used_for_tuning": False, "new_inference_started": False,
    }
    write(RECEIPT, receipt)
    print(json.dumps({"status": "passed", "source_files": len(entries),
                      "cached_contexts": checked["cached_contexts"],
                      "raw_batches": checked["raw_batches"],
                      "audited_methods": 41}), flush=True)


def run_all():
    receipt = read(RECEIPT)
    assert receipt["status"] == "passed" and receipt["new_inference_started"] is False
    assert receipt["source_paused_receipt_sha256"] == sha(PAUSE)
    assert receipt["source_manifest_sha256"] == sha(MANIFEST)
    assert receipt["resume_adapter_sha256"] == sha(SCRIPT)
    assert receipt["frozen_v7_runner_sha256"] == sha(frozen.__file__)
    check_bytes(read(MANIFEST)["files"])
    base, imports = bind(ROOT)
    assert receipt["imported_blocks_freeze_sha256"] == sha(imports)
    assert completed_blocks(base, ROOT) == 41
    assert partial_cache(base, ROOT) == receipt["partial_cache_provenance"]
    print(json.dumps({"status": "resume_gate_passed", "audited_methods": 41,
                      "cached_contexts": 4552, "remaining_contexts": 46784 - 4552}), flush=True)
    faulthandler.enable()
    try:
        frozen.run_all()
    except BaseException as exc:
        try:
            write(ROOT / "DIAGNOSTIC_HOLD.json", {
                "version": "gemma12_v10_isolated_resume_20261001", "status": "failed",
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(), "time": time.time(),
                "test_used_for_tuning": False,
            })
        finally:
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "all"])
    stage = parser.parse_args().stage
    preflight() if stage == "preflight" else run_all()


if __name__ == "__main__":
    main()

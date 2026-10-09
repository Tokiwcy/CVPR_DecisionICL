"""Hash-checked continuation of the user-paused Gemma v10 main-table run.

Only the isolated output root changes. The frozen v7 runner retains the
model, prompts, retrieval, calibration, batching, and target inference.
"""
import argparse
import faulthandler
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import run_gemma12_main_eval_v7_20260928 as frozen


SCRIPT = Path(__file__).resolve()
SOURCE = frozen.ROOT.parent / "evaluation_v10_isolated_resume_20261001"
ROOT = SOURCE.parent / "evaluation_v11_isolated_resume_20261003"
PAUSE = SOURCE / "PAUSED_BY_USER.json"
MANIFEST = ROOT / "resume_source_hashes.json"
RECEIPT = ROOT / "resume_preflight.json"
EXCLUDE = {
    "PAUSED_BY_USER.json", "DIAGNOSTIC_HOLD.json", "WATCHDOG_STOP",
    "status.json", "resume.stdout.log", "resume.stderr.log",
    "resume_launch.json", "resume_source_hashes.json", "resume_preflight.json",
}
COMPLETED = [
    ("aircraft", 1), ("aircraft", 2), ("cub", 1), ("cub", 2),
    ("dogs", 1), ("dogs", 2), ("flowers102", 1),
]
RECHECK = {
    ("cub", 2): "independent_recheck_20261003.json",
    ("dogs", 1): "independent_recheck_20261003.json",
    ("dogs", 2): "independent_recheck_20261002.json",
    ("flowers102", 1): "independent_recheck_20261002.json",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def source_files():
    return sorted(
        (p for p in SOURCE.rglob("*") if p.is_file()
         and p.relative_to(SOURCE).parts[0] not in EXCLUDE),
        key=lambda p: p.relative_to(SOURCE).as_posix(),
    )


def check_bytes(entries):
    assert len(entries) > 60000, "unexpectedly small source state"
    actual = {p.relative_to(SOURCE).as_posix() for p in source_files()}
    assert actual == set(entries), "source file set drift"
    for relative, expected in entries.items():
        src, dst = SOURCE / relative, ROOT / relative
        assert src.is_file() and dst.is_file(), f"missing copy: {relative}"
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
    for dataset, shot in COMPLETED:
        folder = root / f"{shot}shot" / dataset
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
        assert plan_audit["status"] == "passed"
        assert plan_audit["test_labels_used"] is False
        if (dataset, shot) in RECHECK:
            recheck = read(folder / RECHECK[(dataset, shot)])
            assert recheck["status"] == "passed"
            assert recheck["complete_sha256"] == sha(complete)
            assert recheck["audit_sha256"] == sha(folder / "independent_audit.json")
        count += 5 if shot == 1 else 7
    assert count == 53
    partial = root / "2shot" / "flowers102"
    assert not (partial / "complete.json").exists()
    assert not (partial / "independent_audit.json").exists()
    return count


def partial_descriptions(base, root):
    core = base.core
    out = root / "resources" / "flowers102"
    assert core.verified(out / "test_inputs.freeze.json")
    assert not (out / "descriptions.freeze.json").exists()
    receipt = read(PAUSE)
    audit_path = root / "partial_description_audit_20261002.json"
    audit = read(audit_path)
    assert sha(SOURCE / audit_path.name) == receipt["partial_description_audit_sha256"]
    assert sha(audit_path) == receipt["partial_description_audit_sha256"]
    assert audit["status"] == "passed"
    assert audit["full_paired_queries"] == receipt["durable_paired_description_queries"] == 3432
    assert audit["total_queries"] == receipt["expected_description_queries"] == 6149
    assert audit["paired_batches"] == receipt["paired_description_batches"] == 429
    assert audit["file_count"] == 858
    batch_dir = out / "description_batches"
    hashes = audit["file_hashes"]
    assert len(hashes) == 858
    assert {p.name for p in batch_dir.glob("*.json")} == set(hashes)
    for index in range(429):
        for channel in ("attributes", "captions"):
            name = f"{index:05d}.{channel}.json"
            assert sha(batch_dir / name) == hashes[name], name
            item = read(batch_dir / name)
            output = item["output"]
            assert len(item["signature"]) == 64
            assert all(len(output[key]) == 8 for key in
                       ("raw", "texts", "parsed", "generated_lengths"))
    assert sha(batch_dir / "00428.captions.json") == receipt["last_caption_batch_sha256"]
    return {"cached_paired_queries": 3432, "expected_queries": 6149,
            "paired_batches": 429, "batch_files": 858, "complete": False}


def preflight():
    pause = read(PAUSE)
    assert pause["status"] == "paused_by_user"
    assert pause["dataset"] == "flowers102" and pause["shots"] == 2
    assert pause["stage"] == "test_descriptions"
    assert pause["audited_method_results"] == 53
    assert pause["test_used_for_tuning"] is False
    prior = read(SOURCE / "resume_preflight.json")
    assert prior["status"] == "passed"
    assert prior["frozen_v7_runner_sha256"] == sha(frozen.__file__)
    assert sha(SOURCE / "partial_description_audit_20261002.json") == pause["partial_description_audit_sha256"]
    if ROOT.exists():
        assert MANIFEST.is_file() and not RECEIPT.exists(), "v11 root already initialized"
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
    assert completed_blocks(base, ROOT) == 53
    checked = partial_descriptions(base, ROOT)
    result = {
        "version": "gemma12_v11_isolated_resume_20261003", "status": "passed",
        "checked_at_unix": time.time(), "source_paused_receipt_sha256": sha(PAUSE),
        "source_manifest_sha256": sha(MANIFEST), "source_file_count": len(entries),
        "frozen_v7_runner_sha256": sha(frozen.__file__),
        "resume_adapter_sha256": sha(SCRIPT),
        "contract_sha256": sha(ROOT / "contract.json"),
        "run_freeze_sha256": sha(ROOT / "run.freeze.json"),
        "imported_blocks_freeze_sha256": sha(imports),
        "partial_description_provenance": checked,
        "independently_audited_methods_before_resume": 53,
        "test_used_for_tuning": False, "new_inference_started": False,
    }
    write(RECEIPT, result)
    print(json.dumps({"status": "passed", "source_files": len(entries),
                      "cached_paired_queries": 3432, "audited_methods": 53}), flush=True)


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
    assert completed_blocks(base, ROOT) == 53
    assert partial_descriptions(base, ROOT) == receipt["partial_description_provenance"]
    print(json.dumps({"status": "resume_gate_passed", "audited_methods": 53,
                      "cached_paired_queries": 3432,
                      "remaining_description_queries": 6149 - 3432}), flush=True)
    faulthandler.enable()
    try:
        frozen.run_all()
    except BaseException as exc:
        try:
            write(ROOT / "DIAGNOSTIC_HOLD.json", {
                "version": "gemma12_v11_isolated_resume_20261003", "status": "failed",
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

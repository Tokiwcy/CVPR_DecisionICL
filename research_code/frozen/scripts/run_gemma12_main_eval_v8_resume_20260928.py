"""Isolated, hash-checked resume of the paused Gemma v7 main-table run.

The scientific runner is the frozen v7 module. This adapter copies its durable
state to a new root, verifies byte identity and cache/raw provenance, then lets
v7 continue under the same protocol and fixed batch partners. No tuning occurs.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import gemma12_cache_provenance_20260926 as provenance
import run_gemma12_main_eval_v7_20260928 as frozen

SCRIPT = Path(__file__).resolve()
SOURCE = frozen.ROOT
ROOT = SOURCE.parent / "evaluation_v8_isolated_resume_20260928"
MANIFEST = ROOT / "resume_source_hashes.json"
RECEIPT = ROOT / "resume_preflight.json"
EXPECTED = {"cached_contexts": 9326, "expected_contexts": 18947, "raw_batches": 4663}
EXCLUDE = {
    "PAUSED_BY_USER.json", "DIAGNOSTIC_HOLD.json", "WATCHDOG_STOP", "status.json",
    "target_launch_20260928.json", "target_launch_20260928.guard",
    "target_20260928.stdout.log", "target_20260928.stderr.log",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def source_files():
    return sorted((p for p in SOURCE.rglob("*") if p.is_file() and p.relative_to(SOURCE).parts[0] not in EXCLUDE),
                  key=lambda p: p.relative_to(SOURCE).as_posix())


def bind_frozen_runner():
    frozen.ROOT = ROOT
    manifest, imports = frozen.setup()
    base = frozen.base
    assert base.ROOT == ROOT
    assert json.loads(json.dumps(base.CONTRACT)) == read(ROOT / "contract.json"), "scientific contract drift"
    assert base.core.verified(ROOT / "run.freeze.json"), "run freeze drift"
    assert base.core.verified(imports), "import freeze drift"
    assert base.core.verified(ROOT / "1shot" / "aircraft" / "complete.json"), "completed block drift"
    audit = read(ROOT / "1shot" / "aircraft" / "independent_audit.json")
    assert audit.get("status") == "passed" and audit.get("cache_provenance", {}).get("complete") is True
    assert not (ROOT / "2shot" / "aircraft" / "complete.json").exists()
    assert not (ROOT / "2shot" / "aircraft" / "independent_audit.json").exists()
    assert not (ROOT / "PAUSED_BY_USER.json").exists()
    base.cv.verify_tree(ROOT / "2shot" / "aircraft" / "plan.freeze.json")
    return base, imports


def check_copied_bytes(entries):
    assert entries and len(entries) > 9000, "unexpectedly small copied state"
    for relative, expected in entries.items():
        src, dst = SOURCE / relative, ROOT / relative
        assert src.is_file() and dst.is_file(), f"resume source missing: {relative}"
        assert sha(src) == expected and sha(dst) == expected, f"resume source drift: {relative}"


def check_partial_cache(base):
    checked = provenance.audit_cache(base, "aircraft", 2, False)
    assert all(checked.get(key) == value for key, value in EXPECTED.items()), checked
    assert checked.get("complete") is False
    return checked


def preflight():
    pause = read(SOURCE / "PAUSED_BY_USER.json")
    assert pause.get("status") == "paused_by_user" and pause.get("dataset") == "aircraft"
    assert pause.get("shots") == 2 and pause.get("durable_cached_contexts") == EXPECTED["cached_contexts"]
    if ROOT.exists():
        assert MANIFEST.is_file() and not RECEIPT.exists(), "existing resume root is not an incomplete preflight"
        copied = read(MANIFEST)
        assert copied.get("source_root") == str(SOURCE) and copied.get("resume_root") == str(ROOT)
        entries = copied["files"]
        check_copied_bytes(entries)
    else:
        shutil.copytree(SOURCE, ROOT, ignore=lambda directory, names: EXCLUDE & set(names) if Path(directory).resolve() == SOURCE.resolve() else set())
        entries = {p.relative_to(SOURCE).as_posix(): sha(p) for p in source_files()}
        check_copied_bytes(entries)
        write(MANIFEST, {"source_root": str(SOURCE), "resume_root": str(ROOT), "files": entries})
    base, imports = bind_frozen_runner()
    checked = check_partial_cache(base)
    receipt = {
        "version": "gemma12_v8_isolated_resume_20260928", "status": "passed",
        "checked_at_unix": time.time(), "source_paused_receipt_sha256": sha(SOURCE / "PAUSED_BY_USER.json"),
        "source_manifest_sha256": sha(MANIFEST), "source_file_count": len(entries),
        "frozen_v7_runner_sha256": sha(Path(frozen.__file__)), "resume_adapter_sha256": sha(SCRIPT),
        "contract_sha256": sha(ROOT / "contract.json"), "plan_freeze_sha256": sha(ROOT / "2shot" / "aircraft" / "plan.freeze.json"),
        "plan_sha256": sha(ROOT / "2shot" / "aircraft" / "test_plan.json"),
        "imported_blocks_freeze_sha256": sha(imports), "partial_cache_provenance": checked,
        "independently_audited_methods_before_resume": 17,
        "test_used_for_tuning": False, "new_inference_started": False,
    }
    write(RECEIPT, receipt)
    print(json.dumps({"status": "passed", "root": str(ROOT), "source_files": len(entries),
                      "cached_contexts": checked["cached_contexts"], "raw_batches": checked["raw_batches"]}, ensure_ascii=True), flush=True)


def run_all():
    receipt = read(RECEIPT)
    assert receipt.get("status") == "passed" and receipt.get("new_inference_started") is False
    assert receipt.get("resume_adapter_sha256") == sha(SCRIPT), "resume adapter changed"
    assert receipt.get("frozen_v7_runner_sha256") == sha(Path(frozen.__file__)), "frozen runner changed"
    assert receipt.get("source_paused_receipt_sha256") == sha(SOURCE / "PAUSED_BY_USER.json")
    assert receipt.get("source_manifest_sha256") == sha(MANIFEST)
    check_copied_bytes(read(MANIFEST)["files"])
    base, imports = bind_frozen_runner()
    assert receipt.get("imported_blocks_freeze_sha256") == sha(imports)
    assert check_partial_cache(base) == receipt["partial_cache_provenance"]
    print(json.dumps({"status": "resume_gate_passed", "cached_contexts": EXPECTED["cached_contexts"],
                      "remaining_contexts": EXPECTED["expected_contexts"] - EXPECTED["cached_contexts"]}, ensure_ascii=True), flush=True)
    frozen.run_all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "all"])
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight()
    else:
        run_all()


if __name__ == "__main__":
    main()

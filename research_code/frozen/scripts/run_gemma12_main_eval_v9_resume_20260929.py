"""Hash-checked, isolated continuation after the unexpected v8 process exit.

The frozen v7 scientific runner owns every model, prompt, retrieval and result
operation. This adapter only copies durable state, verifies it, and resumes.
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
SOURCE = frozen.ROOT.parent / "evaluation_v8_isolated_resume_20260928"
ROOT = SOURCE.parent / "evaluation_v9_isolated_resume_20260929"
MANIFEST = ROOT / "resume_source_hashes.json"
RECEIPT = ROOT / "resume_preflight.json"
EXCLUDE = {
    "status.json", "resume.stdout.log", "resume.stderr.log", "resume_launch.json",
    "UNEXPECTED_EXIT_20260929.json", "DIAGNOSTIC_HOLD.json", "WATCHDOG_STOP",
    "resume_preflight.json", "resume_source_hashes.json",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def check_bytes(entries):
    assert len(entries) > 10000, "unexpectedly small copied state"
    for relative, expected in entries.items():
        src, dst = SOURCE / relative, ROOT / relative
        assert src.is_file() and dst.is_file(), f"missing copied source: {relative}"
        assert sha(src) == expected and sha(dst) == expected, f"copied bytes drift: {relative}"


def bind():
    frozen.ROOT = ROOT
    _, imports = frozen.setup()
    base = frozen.base
    assert base.ROOT == ROOT
    assert read(ROOT / "contract.json") == json.loads(json.dumps(base.CONTRACT))
    assert base.core.verified(ROOT / "run.freeze.json")
    assert base.core.verified(imports)
    return base, imports


def completed_blocks(base):
    blocks = [("aircraft", 1), ("aircraft", 2), ("cub", 1)]
    for shot in (1, 2):
        frozen.source_block("dtd", shot)
    for ds, shot in blocks:
        folder = ROOT / f"{shot}shot" / ds
        assert base.core.verified(folder / "complete.json"), f"complete drift: {folder}"
        audit = read(folder / "independent_audit.json")
        assert audit["status"] == "passed"
        assert audit["complete_sha256"] == sha(folder / "complete.json")
        assert audit["results_sha256"] == sha(folder / "results.json")
        assert audit["predictions_sha256"] == sha(folder / "predictions.json")
        assert audit["cache_provenance"]["complete"] is True
    assert not (ROOT / "2shot" / "cub" / "complete.json").exists()
    assert not (ROOT / "2shot" / "cub" / "independent_audit.json").exists()
    return len(blocks) + 2


def partial_descriptions(base):
    cv, core = base.cv, base.core
    import run_qwen3_circles_five as circles

    ds = "cub"
    out = ROOT / "resources" / ds
    assert core.verified(out / "test_inputs.freeze.json")
    rows = core.manifest(ds, "test")
    question = core.common.source(ds).instruction
    directory = out / "description_batches"
    actual = {p.name for p in directory.glob("*.json")}
    expected = set()
    full = 0
    partial = []
    batches = list(cv.batches(rows))
    for index, batch in enumerate(batches):
        ap = directory / f"{index:05d}.attributes.json"
        cp = directory / f"{index:05d}.captions.json"
        if not ap.exists():
            assert not cp.exists(), f"caption without attributes: {index}"
            continue
        expected.add(ap.name)
        assert all(core.sha(x.image_path) == x.image_sha256 for x in batch)
        ac = [cv.attribute_conversation(x, question) for x in batch]
        sig = core.stable([core.sha(ROOT / "run.freeze.json"),
                           [(x.sample_id, x.image_sha256) for x in batch], ac])
        ar = read(ap)
        assert ar["signature"] == sig
        ao = ar["output"]
        assert all(len(ao[key]) == len(batch) for key in
                   ("raw", "texts", "parsed", "generated_lengths"))
        attrs = [circles.parse_attribute(text) for text in ao["texts"]]
        assert all(attrs), f"invalid attributes: {index}"
        cc = [cv.caption_conversation(x, a) for x, a in zip(batch, attrs)]
        if not cp.exists():
            partial.append(index)
            continue
        expected.add(cp.name)
        cr = read(cp)
        assert cr["signature"] == core.stable([sig, core.sha(ap), cc])
        co = cr["output"]
        assert all(len(co[key]) == len(batch) for key in
                   ("raw", "texts", "parsed", "generated_lengths"))
        assert all(circles.parse_caption(text) for text in co["texts"]), f"invalid captions: {index}"
        full += len(batch)
    assert actual == expected, "unexpected description batch files"
    assert len(rows) == 4794 and full == 4160 and partial == [520]
    assert len(actual) == 1041 and len(batches) == 600
    assert not (out / "descriptions.freeze.json").exists()
    return {"dataset": ds, "shots": 2, "total_queries": len(rows),
            "full_description_queries": full, "partial_attribute_only_batch": partial,
            "saved_batch_files": len(actual), "batch_size": 8}


def preflight():
    assert SOURCE.is_dir() and (SOURCE / "resume_preflight.json").is_file()
    assert read(SOURCE / "resume_preflight.json")["status"] == "passed"
    assert not (SOURCE / "2shot" / "cub" / "complete.json").exists()
    if ROOT.exists():
        assert MANIFEST.is_file() and RECEIPT.is_file()
        assert read(RECEIPT).get("version") != "gemma12_v9_isolated_resume_20260929", "completed preflight cannot be repeated"
    else:
        shutil.copytree(SOURCE, ROOT,
                        ignore=lambda directory, names: EXCLUDE & set(names)
                        if Path(directory).resolve() == SOURCE.resolve() else set())
    entries = {}
    for path in sorted(SOURCE.rglob("*")):
        if path.is_file() and path.relative_to(SOURCE).parts[0] not in EXCLUDE:
            entries[path.relative_to(SOURCE).as_posix()] = sha(path)
    check_bytes(entries)
    write(MANIFEST, {"source_root": str(SOURCE), "resume_root": str(ROOT), "files": entries})
    base, imports = bind()
    blocks = completed_blocks(base)
    descriptions = partial_descriptions(base)
    receipt = {
        "version": "gemma12_v9_isolated_resume_20260929", "status": "passed",
        "checked_at_unix": time.time(), "source_root": str(SOURCE),
        "resume_root": str(ROOT), "source_manifest_sha256": sha(MANIFEST),
        "source_file_count": len(entries), "frozen_v7_runner_sha256": sha(frozen.__file__),
        "resume_adapter_sha256": sha(SCRIPT), "contract_sha256": sha(ROOT / "contract.json"),
        "run_freeze_sha256": sha(ROOT / "run.freeze.json"),
        "imported_blocks_freeze_sha256": sha(imports),
        "independently_audited_methods_before_resume": 29,
        "verified_completed_blocks": blocks, "partial_descriptions": descriptions,
        "test_used_for_tuning": False, "new_inference_started": False,
    }
    write(RECEIPT, receipt)
    print(json.dumps({"status": "passed", "source_files": len(entries),
                      "audited_methods": 29, **descriptions}, ensure_ascii=True), flush=True)


def run_all():
    receipt = read(RECEIPT)
    assert receipt["status"] == "passed" and receipt["new_inference_started"] is False
    assert receipt["resume_adapter_sha256"] == sha(SCRIPT)
    assert receipt["frozen_v7_runner_sha256"] == sha(frozen.__file__)
    assert receipt["source_manifest_sha256"] == sha(MANIFEST)
    check_bytes(read(MANIFEST)["files"])
    base, imports = bind()
    assert receipt["imported_blocks_freeze_sha256"] == sha(imports)
    assert completed_blocks(base) == receipt["verified_completed_blocks"]
    assert partial_descriptions(base) == receipt["partial_descriptions"]
    print(json.dumps({"status": "resume_gate_passed", "audited_methods": 29,
                      "remaining_cub_description_queries": 634}, ensure_ascii=True), flush=True)
    faulthandler.enable()
    try:
        frozen.run_all()
    except BaseException as exc:
        try:
            write(ROOT / "DIAGNOSTIC_HOLD.json", {
                "version": "gemma12_v9_isolated_resume_20260929", "status": "failed",
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

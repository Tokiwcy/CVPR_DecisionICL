"""Isolated Gemma continuation with eager loading of torchvision image ops.

The frozen v7 scientific runner is unchanged. This process-local preload only
makes the same torchvision decoder available before the model is loaded.
"""
import argparse
import faulthandler
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("TORCHVISION_WARN_WHEN_EXTENSION_LOADING_FAILS", "1")
from torchvision.io import image as tvimage
assert tvimage._has_image_ops(), "torchvision image ops unavailable before model loading"

import run_gemma12_main_eval_v11_resume_20261003 as prior

SCRIPT = Path(__file__).resolve()
SOURCE = prior.ROOT
ROOT = SOURCE.parent / "evaluation_v12_image_extension_preload_20261003"
ORIGINAL_PAUSE = SOURCE.parent / "evaluation_v10_isolated_resume_20261001" / "PAUSED_BY_USER.json"
FAILURE = SOURCE / "DIAGNOSTIC_HOLD.json"
SMOKE = SOURCE / "validation_image_decode_smoke_20261003.json"
MANIFEST = ROOT / "resume_source_hashes.json"
RECEIPT = ROOT / "resume_preflight.json"


def configure():
    prior.SOURCE = SOURCE
    prior.ROOT = ROOT
    prior.PAUSE = ORIGINAL_PAUSE
    prior.MANIFEST = MANIFEST
    prior.RECEIPT = RECEIPT
    prior.SCRIPT = SCRIPT
    prior.RECHECK = {key: "independent_recheck_v12_20261003.json"
                     for key in prior.RECHECK}


def source_failure_and_fix():
    failure = prior.read(FAILURE)
    assert failure["status"] == "failed" and failure["error_type"] == "RuntimeError"
    assert "Couldn't load the image extension" in failure["error"]
    smoke = prior.read(SMOKE)
    assert smoke["status"] == "passed" and smoke["split"] == "validation"
    assert smoke["image_ops_after_model"] is True and smoke["test_access"] is False
    assert smoke["test_used_for_tuning"] is False
    assert tvimage._has_image_ops()
    batch_dir = SOURCE / "resources/flowers102/description_batches"
    assert len(list(batch_dir.glob("*.attributes.json"))) == 429
    assert len(list(batch_dir.glob("*.captions.json"))) == 429
    assert not (SOURCE / "resources/flowers102/descriptions.freeze.json").exists()
    assert not (SOURCE / "2shot/flowers102/complete.json").exists()
    return prior.sha(FAILURE), prior.sha(SMOKE)


def preflight():
    failure_sha, smoke_sha = source_failure_and_fix()
    configure()
    prior.preflight()
    result = prior.read(RECEIPT)
    assert result["status"] == "passed" and result["source_file_count"] > 83000
    assert result["independently_audited_methods_before_resume"] == 53
    assert result["partial_description_provenance"]["cached_paired_queries"] == 3432
    result.update(version="gemma12_v12_image_extension_preload_20261003",
                  source_failure_receipt_sha256=failure_sha,
                  validation_image_decode_smoke_sha256=smoke_sha,
                  torchvision_image_ops_preloaded=True,
                  resume_adapter_sha256=prior.sha(SCRIPT))
    prior.write(RECEIPT, result)
    print(json.dumps({"status": "passed", "source_files": result["source_file_count"],
                      "cached_paired_queries": 3432, "audited_methods": 53,
                      "image_ops_preloaded": True}), flush=True)


def run_all():
    failure_sha, smoke_sha = source_failure_and_fix()
    configure()
    receipt = prior.read(RECEIPT)
    assert receipt["version"] == "gemma12_v12_image_extension_preload_20261003"
    assert receipt["status"] == "passed" and receipt["new_inference_started"] is False
    assert receipt["source_failure_receipt_sha256"] == failure_sha
    assert receipt["validation_image_decode_smoke_sha256"] == smoke_sha
    assert receipt["source_paused_receipt_sha256"] == prior.sha(ORIGINAL_PAUSE)
    assert receipt["source_manifest_sha256"] == prior.sha(MANIFEST)
    assert receipt["resume_adapter_sha256"] == prior.sha(SCRIPT)
    assert receipt["frozen_v7_runner_sha256"] == prior.sha(prior.frozen.__file__)
    prior.check_bytes(prior.read(MANIFEST)["files"])
    base, imports = prior.bind(ROOT)
    assert receipt["imported_blocks_freeze_sha256"] == prior.sha(imports)
    assert prior.completed_blocks(base, ROOT) == 53
    assert prior.partial_descriptions(base, ROOT) == receipt["partial_description_provenance"]
    assert tvimage._has_image_ops()
    print(json.dumps({"status": "resume_gate_passed", "audited_methods": 53,
                      "cached_paired_queries": 3432,
                      "remaining_description_queries": 6149 - 3432,
                      "image_ops_preloaded": True}), flush=True)
    faulthandler.enable()
    try:
        prior.frozen.run_all()
    except BaseException as exc:
        try:
            prior.write(ROOT / "DIAGNOSTIC_HOLD.json", {
                "version": "gemma12_v12_image_extension_preload_20261003",
                "status": "failed", "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(),
                "time": time.time(), "test_used_for_tuning": False,
            })
        finally:
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "all"])
    args = parser.parse_args()
    preflight() if args.stage == "preflight" else run_all()


if __name__ == "__main__":
    main()

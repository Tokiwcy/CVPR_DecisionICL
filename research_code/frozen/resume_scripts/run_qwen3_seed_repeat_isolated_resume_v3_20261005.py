"""Resume frozen Qwen3 runner from a CPU-audited isolated cache.

Only the runtime memory ceiling is raised from 0.88 to validated 0.94.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT = Path(r"D:\BehaviorICL")
SCRIPTS = PROJECT / "decision_guided_retrieval_cleanroom/scripts"
sys.path.insert(0, str(SCRIPTS))
import run_qwen3_main_three_seed_full_20261004 as run

ROOT = PROJECT / "decision_guided_retrieval_cleanroom/artifacts/qwen3_main_three_seed_full_v3_isolated_resume_20261005"
SOURCE = PROJECT / "decision_guided_retrieval_cleanroom/artifacts/qwen3_main_three_seed_full_20261004"
ORIGINAL_CAP = torch.cuda.set_per_process_memory_fraction


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_cap(requested: float, device=None):
    if requested != 0.88:
        raise RuntimeError(f"unexpected frozen memory cap request: {requested}")
    return ORIGINAL_CAP(0.94, device)


def main() -> None:
    receipt = json.loads((ROOT / "resume_v3_preflight.json").read_text(encoding="utf-8"))
    if receipt["status"] != "passed" or receipt["destination_root"] != str(ROOT):
        raise RuntimeError("isolated CPU preflight not passed")
    if sha(SOURCE / "PAUSED_BY_USER.json") != receipt["source_pause_sha256"]:
        raise RuntimeError("source pause receipt changed")
    cache = ROOT / "seed17/2shot/dogs/context_cache.jsonl"
    if sha(cache) != receipt["dogs_2shot_prefix_sha256"]:
        raise RuntimeError("isolated cache prefix changed")
    if any((ROOT / name).exists() for name in ("PAUSED_BY_USER.json", "DIAGNOSTIC_HOLD.json", "WATCHDOG_STOP")):
        raise RuntimeError("isolated run held")
    torch.cuda.set_per_process_memory_fraction = memory_cap
    run.ROOT = ROOT
    run.run()


if __name__ == "__main__":
    main()

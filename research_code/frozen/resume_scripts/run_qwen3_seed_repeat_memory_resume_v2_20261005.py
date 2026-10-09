"""Runtime-only Qwen3 seed repeat recovery after a reproducible batch2 VRAM cap OOM.

The frozen science runner, plan, batch partner, model, prompt and decoding are unchanged.
Only the PyTorch per-process VRAM ceiling rises from 0.88 to 0.94.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch

RUNNER = Path(r"D:\BehaviorICL\decision_guided_retrieval_cleanroom\scripts\run_qwen3_main_three_seed_full_20261004.py")
ORIGINAL = torch.cuda.set_per_process_memory_fraction


def memory_only_cap(requested: float, device=None):
    if requested != 0.88:
        raise RuntimeError(f"unexpected Qwen3 memory cap request: {requested}")
    return ORIGINAL(0.94, device)


torch.cuda.set_per_process_memory_fraction = memory_only_cap
sys.argv = [str(RUNNER), "all"]
runpy.run_path(str(RUNNER), run_name="__main__")

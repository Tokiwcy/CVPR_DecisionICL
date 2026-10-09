"""Isolated, fresh-target Qwen3 main-table repeats for seeds 17 and 73."""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import msvcrt
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import run_qwen3_five_dataset_recalibrated as r
import run_qwen3_gpu_accelerated_queue as q
import run_qwen3_flowers_replacement_queue as flowers  # patches the frozen Flowers source only

ROOT = r.PROJECT / "artifacts/qwen3_main_three_seed_full_20261004"
DOC = r.PROJECT / "docs/qwen3_main_three_seed_full_20261004_zh.md"
PRIOR = r.PROJECT / "artifacts/qwen3_five_dataset_recalibrated_20260910"
ORDER = ("dtd", "aircraft", "cub", "dogs", "flowers102")
SEEDS = (17, 73)
OLD_FIT = r.fit
OLD_SOURCES = r.source_files
OLD_INFER_CONTRACT = r.infer_contract
OLD_RESOURCES = {
    **{d: PRIOR / "gpu_batch2/resources" for d in ORDER if d != "flowers102"},
    "flowers102": PRIOR / "gpu_batch2_flowers102_20260911/resources",
}
OLD_RESOURCES["dtd"] = PRIOR / "resources"


def write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def sources():
    return OLD_SOURCES() + [Path(__file__), DOC]


def held():
    for path in (ROOT, r.ROOT):
        if any((path / name).exists() for name in ("PAUSED_BY_USER.json", "DIAGNOSTIC_HOLD.json", "WATCHDOG_STOP")):
            raise RuntimeError("seed repeat held by control file: " + str(path))


def configure(seed: int, dataset: str, shots: int) -> Path:
    if seed not in SEEDS or dataset not in ORDER or shots not in (1, 2):
        raise ValueError("outside frozen seed repeat scope")
    root = ROOT / f"seed{seed}"
    r.ROOT = root
    r.RESOURCES = OLD_RESOURCES[dataset]
    r.ORDER = ORDER
    r.source_files = sources
    r.held = held
    r.migrate = lambda *args, **kwargs: []
    r.update_evidence = lambda *args, **kwargs: None
    r.__dict__["SEED"] = seed
    q.FAST = root
    q.CURRENT.update(dataset=dataset, shots=shots, fast=(dataset, shots) != ("dtd", 1))
    r.infer_contract = OLD_INFER_CONTRACT if (dataset, shots) == ("dtd", 1) else q.fast_contract
    return root


def install_seeded_fit(seed: int) -> None:
    """Change only the two seed literals in the original frozen fit source."""
    src = inspect.getsource(OLD_FIT)
    replacements = {
        "def fit(dataset, shots):": "def seeded_fit(dataset, shots):",
        "folds=5, seed=42, shots=shots,": "folds=5, seed=SEED, shots=shots,",
        "rng = np.random.default_rng(42)": "rng = np.random.default_rng(SEED)",
    }
    for old, new in replacements.items():
        if src.count(old) != 1:
            raise ValueError("frozen fit source differs: " + old)
        src = src.replace(old, new)
    namespace: dict[str, object] = {}
    exec(compile(src, str(Path(__file__).resolve()) + ":seeded_fit", "exec"), r.__dict__, namespace)
    r.fit = namespace["seeded_fit"]

    def fit_class_equal(features, validation_y, validation):
        folds = r.official.stratified_folds(validation, 5, seed)
        oof = r.np.empty(len(validation), r.np.int64)
        history = []
        for fold in range(5):
            fit = r.np.flatnonzero(folds != fold)
            hold = r.np.flatnonzero(folds == fold)
            alpha, score = r.common.fit_alpha(features, validation_y, fit)
            weights = r.common.cub.shared.equal_weights(features.shape[2], alpha)
            oof[hold] = r.score_classes(features[hold], weights).argmax(1)
            history.append(dict(fold=fold, alpha=alpha, weights=weights.tolist(),
                                fit_class_accuracy=score, fit_count=len(fit), heldout_count=len(hold)))
        alpha, score = r.common.fit_alpha(features, validation_y, r.np.arange(len(validation)))
        weights = r.common.cub.shared.equal_weights(features.shape[2], alpha)
        return oof, alpha, weights, score, history

    r.official.fit_class_equal = fit_class_equal


def preflight() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    r.ORDER = ORDER
    r.source_files = sources
    for seed in SEEDS:
        configure(seed, "dtd", 1)
        install_seeded_fit(seed)
        if not (r.ROOT / "input_audit.json").exists():
            r.audit_inputs()
        else:
            r.assert_audit()
        prior = [PRIOR / "gpu_batch2/validation_smoke.json",
                 PRIOR / "gpu_batch2_flowers102_20260911/validation_smoke.json"]
        if not all(path.is_file() for path in prior):
            raise FileNotFoundError("original validation-only smoke missing")
        write(r.ROOT / "validation_smoke.json", dict(status="passed", validation_only=True,
              original_smoke_sha256=r.hashes(prior), source_files=r.hashes(sources()),
              model_and_prompt_unchanged=True, test_used_for_tuning=False))
        for dataset in ORDER:
            for shots in (1, 2):
                configure(seed, dataset, shots)
                resources = r.RESOURCES / dataset
                names = ["clip.train.npz", "clip.validation.npz", "clip.test.npz"]
                if shots == 2:
                    names += ["clip.captions.validation.npz", "clip.captions.test.npz",
                              "counterfactual_descriptions.validation.jsonl",
                              "counterfactual_descriptions.test.jsonl"]
                if not all((resources / name).is_file() for name in names):
                    raise FileNotFoundError("frozen resource missing: " + str(resources))
                r.fit(dataset, shots)
                r.verify_plan(dataset, shots)
                r.emit("seed_plan_preflight_passed", seed=seed, dataset=dataset, shots=shots)
    write(ROOT / "prelaunch.json", dict(status="passed", seeds=list(SEEDS), datasets=list(ORDER),
          shots=[1, 2], expected_new_method_items=120, prior_seed=42,
          target_predictions_reused=False, test_used_for_tuning=False,
          frozen_sources=r.hashes(sources())))


def run() -> None:
    receipt = ROOT / "prelaunch.json"
    if not receipt.exists() or json.loads(receipt.read_text(encoding="utf-8"))["status"] != "passed":
        raise RuntimeError("independent prelaunch gate missing")
    # Original main queue and cleanroom queue are both used by other experiments.
    locks = [PRIOR / "gpu_queue.lock", r.PROJECT / "artifacts/priority_1shot_ablation_20260920/gpu_queue.lock"]
    handles = []
    try:
        for path in sorted(locks):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+b")
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            handles.append(handle)
        for seed in SEEDS:
            configure(seed, "dtd", 1)
            install_seeded_fit(seed)
            r.assert_audit()
            for dataset in ORDER:
                for shots in (1, 2):
                    configure(seed, dataset, shots)
                    held()
                    r.verify_plan(dataset, shots)
                    out = r.ROOT / f"{shots}shot" / dataset
                    if (out / "complete.json").exists():
                        r.verify_complete(dataset, shots)
                        continue
                    backend = r.official.OfficialBackend() if (dataset, shots) == ("dtd", 1) else q.FixedBatchBackend(dataset, shots)
                    try:
                        r.infer(dataset, shots, backend)
                    finally:
                        del backend
                        gc.collect()
                        import torch
                        torch.cuda.empty_cache()
        write(ROOT / "target_complete_pending_independent_audit.json", dict(
            seeds=list(SEEDS), completed_blocks=20, target_predictions_reused=False,
            test_used_for_tuning=False, time=time.time()))
    finally:
        for handle in reversed(handles):
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "all"))
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight()
    else:
        run()


if __name__ == "__main__":
    main()

"""User-selected no-debias ablation: independent calibration before test access."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
import numpy as np
import torch
import run_cub_full_current_atoms as base
import run_cub_atom_label_residual as shared
from dgdr.atom_label import build_class_atom_prototypes, fit_nonnegative_residual, score_classes
from dgdr.calibration import stratified_folds
from dgdr.evidence import EmbeddingStore, EvidenceStore
from dgdr.hashing import file_sha256
from dgdr.io import load_atom_bank, load_manifest, read_json, write_json
from raw_siglip_evidence import POLICY, validate_raw_policy
from run_fgclip2_evidence_comparison import raw_store, check, freeze, verify
from report_full_suite_significance import holm

DATA = PROJECT / "artifacts/cub_full_current_atoms_20260904"
RAW = PROJECT / "artifacts/siglip_raw_selected_20260906/cub"
OLD = PROJECT / "artifacts/five_dataset_full_suite_20260905/cub"
OUT = PROJECT / "artifacts/cub_raw_siglip_main_methods_20260906"
PROTOCOL = PROJECT / "docs/cub_raw_main_methods_20260906_zh.md"
METHODS = ["rices", "muier", "atom_label_equal_alpha_raw", "atom_label_residual_raw",
           "rices_atom_residual_raw", "nnls_raw", "equal_alpha"]
PARENTS = {m: ("muier" if m.startswith("atom_label_") else "rices")
           for m in METHODS if m not in ("rices", "muier")}
BANK_HASH = "5e7c43f63645ac50d2c1da6caa9243ca25cb9ebc078abd6d881f6893c6969c81"


def emit(stage, **kw):
    print(json.dumps(dict(stage=stage, **kw)), flush=True)


def paired_files(prefix):
    return [Path(str(prefix) + "." + ext) for ext in ("json", "npz")]


def validate_alignment(samples, embedding, evidence, bank):
    check(evidence, bank, samples)
    validate_raw_policy(evidence)
    if embedding.sample_ids != evidence.sample_ids:
        raise ValueError("embedding/evidence order drift")
    if (embedding.model_id, embedding.model_revision) != (evidence.model_id, evidence.model_revision):
        raise ValueError("encoder contract drift")
    if not np.isfinite(embedding.embeddings).all():
        raise ValueError("nonfinite embedding")


def fit_image_methods(features, val_labels, train_labels, rows, grams, crosses):
    alpha, _ = base.select_alpha(features, rows, val_labels, train_labels)
    return {"equal_alpha": base.equal_weights(features.shape[2], alpha),
            "nnls_raw": base.fit_nnls(grams, crosses, rows)}


def fit_class_methods(features, val_labels, rows, parent):
    if parent == "rices":
        return {"rices_atom_residual_raw": fit_nonnegative_residual(features, val_labels, rows)}
    alpha, _ = shared.select_alpha(features, val_labels, rows)
    return {"atom_label_equal_alpha_raw": shared.equal_weights(features.shape[2], alpha),
            "atom_label_residual_raw": fit_nonnegative_residual(features, val_labels, rows)}


def calibrate():
    if (OUT / "calibration.freeze.json").exists():
        verify(OUT / "calibration.freeze.json")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    bank = load_atom_bank(RAW / "atom_bank.original.json")
    if bank.bank_hash != BANK_HASH:
        raise ValueError("not the original CUB bank")
    sources = [RAW / "atom_bank.original.json", PROTOCOL, Path(__file__),
               Path(__file__).with_name("raw_siglip_evidence.py"),
               Path(__file__).with_name("run_fgclip2_evidence_comparison.py"),
               Path(__file__).with_name("run_cub_full_current_atoms.py"),
               Path(__file__).with_name("run_cub_atom_label_residual.py"),
               Path(__file__).with_name("run_full_suite_target.py"),
               Path(__file__).with_name("run_qwen3_dataset_plan.py"),
               Path(__file__).with_name("report_full_suite_significance.py")]
    sources += sorted((PROJECT / "src/dgdr").rglob("*.py"))
    samples, emb, ev = {}, {}, {}
    for split in ("train", "validation"):
        mp = DATA / f"cub.{split}.jsonl"
        samples[split] = load_manifest(mp, expected_split=split)
        emb[split] = EmbeddingStore.load(DATA / f"siglip.{split}.embeddings")
        ev[split] = EvidenceStore.load(RAW / f"siglip.{split}.evidence")
        validate_alignment(samples[split], emb[split], ev[split], bank)
        original = EvidenceStore.load(DATA / f"siglip.{split}.evidence")
        np.testing.assert_array_equal(ev[split].raw_logits, original.raw_logits)
        rebuilt = raw_store(original, bank)
        for field in ("state_probabilities", "signed_margins", "ordinal_scores", "applicability_scores"):
            np.testing.assert_array_equal(getattr(ev[split], field), getattr(rebuilt, field))
        sources += [mp] + paired_files(DATA / f"siglip.{split}.embeddings")
        sources += paired_files(RAW / f"siglip.{split}.evidence") + paired_files(DATA / f"siglip.{split}.evidence")
    if [len(samples[s]) for s in ("train", "validation")] != [5993, 1000]:
        raise ValueError("original full CUB split drift")
    for rows in samples.values():
        for sample in rows:
            if file_sha256(Path(sample.image_path)) != sample.image_sha256:
                raise ValueError("image changed: " + sample.sample_id)
    contract = dict(policy=POLICY, methods=METHODS, target="Qwen3-VL-4B-Instruct", shots=1,
                    bank_hash=bank.bank_hash, inputs={str(p): file_sha256(p) for p in sources},
                    test_used=False, inherited_weights=False)
    cp = OUT / "input_contract.json"
    if cp.exists() and read_json(cp) != contract:
        raise ValueError("input/source contract changed; use a new output directory")
    write_json(cp, contract)
    labels, _, train_y, val_y = shared.label_contract(samples["train"], samples["validation"])
    label_embeddings = shared.encode_label_texts(labels)
    np.save(OUT / "label_embeddings.npy", label_embeddings)
    write_json(OUT / "label_order.json", list(labels))
    gc.collect()
    torch.cuda.empty_cache()
    prototypes = build_class_atom_prototypes(samples=samples["train"], evidence=ev["train"])
    if prototypes.labels != labels:
        raise ValueError("class prototype label mismatch")
    folds = stratified_folds(samples["validation"], 5, 42)
    train_ids = np.asarray(emb["train"].sample_ids)
    val_plans, full, histories = {}, {}, {m: [] for m in PARENTS}
    features = base.feature_tensor(emb["validation"], emb["train"], ev["validation"], ev["train"])
    grams, crosses = base.nnls_statistics(features, val_y, train_y)
    baseline = features[:, :, 0].argmax(1)
    val_plans["rices"] = train_ids[baseline].tolist()
    oof = {m: np.empty(len(val_y), dtype=np.int64) for m in ("equal_alpha", "nnls_raw")}
    for fold in range(5):
        fit, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        weights = fit_image_methods(features, val_y, train_y, fit, grams, crosses)
        for m, w in weights.items():
            oof[m][held] = base.top_indices(features[held], w)
            histories[m].append(dict(heldout=fold, fit_count=len(fit), weights=w.tolist()))
        emit("image_fold", fold=fold)
    full.update(fit_image_methods(features, val_y, train_y, np.arange(len(val_y)), grams, crosses))
    val_plans.update({m: train_ids[v].tolist() for m, v in oof.items()})
    del features, grams, crosses
    gc.collect()
    for parent, texts in (("muier", label_embeddings), ("rices", np.zeros_like(label_embeddings))):
        features, candidates = shared.class_features(emb["validation"], ev["validation"],
            emb["train"], train_y, texts, prototypes)
        baseline = shared.classes_to_demo_ids(features[:, :, 0].argmax(1), candidates, train_ids)
        if parent in val_plans and val_plans[parent] != baseline:
            raise ValueError("full-pool class-max baseline mismatch")
        val_plans[parent] = baseline
        methods = [m for m in PARENTS if m.startswith("atom_label_")] if parent == "muier" else ["rices_atom_residual_raw"]
        oof = {m: np.empty(len(val_y), dtype=np.int64) for m in methods}
        for fold in range(5):
            fit, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            for m, w in fit_class_methods(features, val_y, fit, parent).items():
                oof[m][held] = score_classes(features[held], w).argmax(1)
                histories[m].append(dict(heldout=fold, fit_count=len(fit), weights=w.tolist()))
        full.update(fit_class_methods(features, val_y, np.arange(len(val_y)), parent))
        val_plans.update({m: shared.classes_to_demo_ids(v, candidates, train_ids) for m, v in oof.items()})
        emit("class_calibrated", parent=parent)
    train_truth = {s.sample_id: s.label for s in samples["train"]}
    metrics = {m: float(np.mean([train_truth[d] == q.label for q, d in zip(samples["validation"], ds)]))
               for m, ds in val_plans.items()}
    calibration = dict(bank_hash=bank.bank_hash, policy=POLICY, folds=folds.tolist(),
        full_weights={m: w.tolist() for m, w in full.items()}, outer_folds=histories,
        oof_same_class_accuracy=metrics, test_labels_used=False, inherited_weights=False,
        selected_deployment={m: m if metrics[m] > metrics[p] + 1e-12 else p for m, p in PARENTS.items()},
        training_objective="historical same-class demo affinity; not target accuracy", alpha_grid=base.ALPHAS.tolist())
    write_json(OUT / "calibration.json", calibration)
    write_json(OUT / "validation_plan.json", dict(query_ids=list(emb["validation"].sample_ids), methods=val_plans))
    freeze(OUT / "calibration.freeze.json", sources + [cp, OUT / "calibration.json",
        OUT / "validation_plan.json", OUT / "label_embeddings.npy", OUT / "label_order.json"])
    emit("calibration_frozen", oof=metrics, deployment=calibration["selected_deployment"])


def prepare_test():
    verify(OUT / "calibration.freeze.json")
    if (OUT / "preparation_complete.json").exists():
        verify(OUT / "preparation_complete.json")
        return
    bank = load_atom_bank(RAW / "atom_bank.original.json")
    train = load_manifest(DATA / "cub.train.jsonl", expected_split="train")
    te = EmbeddingStore.load(DATA / "siglip.train.embeddings")
    tv = EvidenceStore.load(RAW / "siglip.train.evidence")
    qe = EmbeddingStore.load(DATA / "siglip.test.embeddings")
    original = EvidenceStore.load(DATA / "siglip.test.evidence")
    qv = raw_store(original, bank)
    validate_raw_policy(qv)
    if qv.bank_hash != bank.bank_hash or qv.sample_ids != qe.sample_ids or len(qe.sample_ids) != 4794:
        raise ValueError("test bank/order/count drift")
    np.testing.assert_array_equal(qv.raw_logits, original.raw_logits)
    qv.save(OUT / "siglip.test.evidence")
    labels, _, ty, _ = shared.label_contract(train, [])
    if list(labels) != read_json(OUT / "label_order.json"):
        raise ValueError("label embedding order drift")
    texts = np.load(OUT / "label_embeddings.npy")
    prototypes = build_class_atom_prototypes(samples=train, evidence=tv)
    cal = read_json(OUT / "calibration.json")
    weights = {m: np.asarray(w, np.float32) for m, w in cal["full_weights"].items()}
    methods = {m: [] for m in METHODS}
    tids = np.asarray(te.sample_ids)
    for start in range(0, len(qe.sample_ids), 64):
        stop = min(start + 64, len(qe.sample_ids))
        eb = EmbeddingStore(qe.sample_ids[start:stop], qe.embeddings[start:stop], qe.model_id, qe.model_revision)
        ev = shared.sliced_evidence(qv, start, stop)
        ft = base.feature_tensor(eb, te, ev, tv)
        methods["rices"].extend(tids[ft[:, :, 0].argmax(1)].tolist())
        for m in ("nnls_raw", "equal_alpha"):
            methods[m].extend(tids[base.top_indices(ft, weights[m])].tolist())
        for parent, lt in (("muier", texts), ("rices", np.zeros_like(texts))):
            cf, best = shared.class_features(eb, ev, te, ty, lt, prototypes)
            baseline = shared.classes_to_demo_ids(cf[:, :, 0].argmax(1), best, tids)
            if parent == "muier":
                methods[parent].extend(baseline)
                names = ("atom_label_equal_alpha_raw", "atom_label_residual_raw")
                direct = (eb.embeddings @ te.embeddings.T + (eb.embeddings @ texts.T)[:, ty]).argmax(1)
                if baseline != tids[direct].tolist():
                    raise ValueError("MUIER class-max is not full-pool equivalent")
            else:
                if baseline != methods["rices"][start:stop]:
                    raise ValueError("RICES class-max parity failed")
                names = ("rices_atom_residual_raw",)
            for m in names:
                methods[m].extend(shared.classes_to_demo_ids(score_classes(cf, weights[m]).argmax(1), best, tids))
        if stop % 512 == 0 or stop == len(qe.sample_ids):
            emit("test_plan", completed=stop, total=len(qe.sample_ids))
    old = read_json(OLD / "test_plan.json")
    if old["query_ids"] != list(qe.sample_ids):
        raise ValueError("historical test order mismatch")
    for m in ("rices", "muier"):
        if methods[m] != old["methods"][m]:
            raise ValueError("historical baseline parity failed: " + m)
    train_ids = set(tids)
    for m, ds in methods.items():
        if len(ds) != 4794 or any(d not in train_ids for d in ds):
            raise ValueError("invalid complete demo plan " + m)
    plan = dict(dataset="cub", split="test", shots=1, query_ids=list(qe.sample_ids), methods=methods,
        active_target_methods=METHODS, skipped_target_methods={}, candidate_pool="complete train split",
        test_labels_used_to_create_plan=False, calibration_sha256=file_sha256(OUT / "calibration.json"),
        evidence_policy=POLICY, selected_deployment=cal["selected_deployment"])
    write_json(OUT / "test_plan.json", plan)
    write_json(OUT / "test_plan.freeze.json", dict(sha256=file_sha256(OUT / "test_plan.json")))
    sources = [DATA / "cub.test.jsonl", OLD / "test_plan.json", OUT / "calibration.freeze.json",
               OUT / "test_plan.json", OUT / "test_plan.freeze.json"]
    for prefix in (DATA / "siglip.test.embeddings", DATA / "siglip.test.evidence", OUT / "siglip.test.evidence"):
        sources += paired_files(prefix)
    freeze(OUT / "preparation_complete.json", sources)
    emit("preparation_complete", methods=METHODS, baseline_parity=True)


def report():
    verify(OUT / "preparation_complete.json")
    result = read_json(OUT / "qwen3_results.json")
    if result["status"] != "complete":
        raise ValueError("target inference not complete")
    pred = read_json(OUT / "qwen3_predictions.json")
    old_pred = read_json(OLD / "qwen3_predictions.json")
    from run_qwen3_dataset_plan import paired
    truth = {s.sample_id: s.label for s in load_manifest(DATA / "cub.test.jsonl", expected_split="test")}
    for m in METHODS:
        if set(pred[m]) != set(truth):
            raise ValueError("prediction coverage mismatch")
        assert abs(np.mean([pred[m][q] == y for q, y in truth.items()]) - result["methods"][m]["accuracy"]) < 1e-12
    primary = {m: paired(pred[p], pred[m], truth) for m, p in PARENTS.items()}
    for m, adjusted in zip(PARENTS, holm([primary[m]["exact_mcnemar_p"] for m in PARENTS])):
        primary[m]["holm_five_methods"] = adjusted
    debias = {m: paired(old_pred[m], pred[m], truth) for m in PARENTS}
    write_json(OUT / "comparison_report.json", dict(primary=primary, raw_vs_original_debiased=debias,
        exploratory=True, metrics=result["methods"], selected_deployment=read_json(OUT / "calibration.json")["selected_deployment"]))
    lines = ["# CUB 原 atom 无去偏：完整目标结果", "", "Qwen3-VL-4B-Instruct；1-shot / batch 1；4794 test 图。全部权重独立重学；CUB 为探索性数据集。", "",
             "| 方法 | Accuracy | Macro | 对父基线救回/误伤 | p / Holm-5 |", "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        met, pair = result["methods"][m], primary.get(m)
        counts = f"{pair['rescue']}/{pair['harm']}" if pair else "—"
        p = f"{pair['exact_mcnemar_p']:.4g} / {pair['holm_five_methods']:.4g}" if pair else "—"
        lines.append(f"| {m} | {100*met['accuracy']:.3f}% | {100*met['macro_accuracy']:.3f}% | {counts} | {p} |")
    lines += ["", "父基线按预声明分别为 RICES 或 MUIER；所有五个 raw 变体都报告，不按 test 选择部署。OOF fallback 规则见 calibration.json。图级 bootstrap CI 和逐方法去偏/无去偏配对见 comparison_report.json；不是类别聚类 CI。", "",
              "本轮只检验检索与目标分类效果，不证明原 atom 的语义、覆盖或视觉分数已经修好。旧队列仍暂停。"]
    (OUT / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    freeze(OUT / "complete.json", [OUT / "preparation_complete.json", OUT / "qwen3_results.json",
        OUT / "qwen3_predictions.json", OUT / "qwen3_inference_protocol.json", OUT / "comparison_report.json",
        OUT / "report_zh.md", OLD / "qwen3_predictions.json"])
    emit("complete", metrics=result["methods"], primary=primary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "target", "report", "all"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage in ("prepare", "all"):
            calibrate()
            prepare_test()
        if args.stage in ("target", "all"):
            verify(OUT / "preparation_complete.json")
            subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("run_full_suite_target.py")),
                "--dataset", "cub", "--output-dir", str(OUT), "--reuse-dir", str(OLD)], check=True)
        if args.stage in ("report", "all"):
            report()
        write_json(OUT / "runner_status.json", dict(stage=args.stage, state="complete", updated_at=time.time()))
    except Exception as exc:
        write_json(OUT / "runner_status.json", dict(stage=args.stage, state="failed", error=repr(exc), updated_at=time.time()))
        raise


if __name__ == "__main__":
    main()

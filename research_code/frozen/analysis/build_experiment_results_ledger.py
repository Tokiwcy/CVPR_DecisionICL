"""Build a focused experiment ledger: main tables and authorized ablations.

This is an aggregation/receipt-verification script; it never runs model inference.
Historical reports and completed secondary panels remain in source artifacts but are
intentionally omitted from this focused view.
"""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "decision_guided_retrieval_cleanroom"
ART = CLEAN / "artifacts"
OUT = ROOT / "docs/experiment_results_ledger_20260920_zh.md"
SOURCES = {}
LINES = []
FLAT = []
PAIRED = []
MAIN_ROWS = []
GENERIC_ROWS = []
ROTATED_ROWS = []
VISUAL_CENTROID_ROWS = []
PARENT_GATE_ROWS = []
SEED_REPEAT_ROWS = []

DATASETS = ["dtd", "aircraft", "cub", "dogs", "flowers102"]
NAMES = ["DTD", "Aircraft", "CUB", "Dogs", "Flowers102"]
MAIN_KEYS = [
    "random_official_prompt",
    "rices_clip_official_prompt",
    "muier_classification_adapted_siglip_official_prompt",
    "rices_clip_atom_class_equal_raw_official_prompt",
    "muier_classification_adapted_siglip_atom_class_equal_raw_official_prompt",
    "circles_official_prompt",
    "circles_atom_class_equal_raw_official_prompt",
]
MAIN_LABELS = ["Random", "RICES", "MUIER", "RICES+Decision", "MUIER+Decision", "CIRCLES", "CIRCLES+Decision"]
GENERIC_KEYS = ["rices+generic_visual_direct_sample", "muier+generic_visual_direct_sample", "circles+generic_visual_direct_sample"]
GEMMA_METHODS = ["random", "rices", "muier", "rices+matched_decision", "muier+matched_decision", "circles", "circles+matched_decision"]


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(raw).hexdigest()
    return raw.decode("utf-8-sig")


def js(path):
    return json.loads(read(path))


def link(path, label="原始记录"):
    return f"[{label}](</{Path(path).as_posix()}>)"


def add(text=""):
    LINES.append(text)


def table(headers, rows):
    add("| " + " | ".join(headers) + " |")
    add("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        add("| " + " | ".join(map(str, row)) + " |")
    add()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_complete(path):
    obj = js(path)
    files = obj.get("files")
    assert isinstance(files, dict) and files, f"empty complete receipt: {path}"
    for name, expected in files.items():
        candidate = Path(name)
        assert candidate.is_file(), f"complete member missing: {candidate}"
        assert sha(candidate) == expected, f"complete member hash mismatch: {candidate}"
    return obj


def add_flat(model, dataset, shot, method, item, source):
    metrics = item.get("metrics", {})
    accuracy = metrics.get("accuracy")
    if accuracy is None:
        return
    FLAT.append({
        "model": model, "dataset": dataset, "shots": shot, "method": method,
        "accuracy": accuracy, "macro_accuracy": metrics.get("macro_accuracy"),
        "invalid_outputs": item.get("invalid_exact_output_count"), "source": str(source.relative_to(ROOT)).replace("\\", "/")
    })


def pct(value):
    return f"{100 * value:.3f}" if value is not None else "待完成"


def record_pair(model, dataset, shot, method, comparator, pair, source):
    ci = pair.get("paired_bootstrap_95pct") or pair.get("paired_bootstrap_95") or [None, None]
    PAIRED.append({
        "model": model, "dataset": dataset, "shots": shot,
        "method": method, "comparator": comparator,
        "delta_pp": 100 * pair["accuracy_delta"],
        "rescue": pair.get("rescue"), "harm": pair.get("harm"),
        "ci_low_pp": 100 * ci[0] if ci[0] is not None else None,
        "ci_high_pp": 100 * ci[1] if ci[1] is not None else None,
        "exact_mcnemar_p": pair.get("exact_mcnemar_p"),
        "source": str(source.relative_to(ROOT)).replace("\\", "/")
    })


def main_model(title, folder):
    path = ART / folder / "live_results.json"
    obj = js(path)
    results = obj.get("results", {})
    count = 0
    for shot in (1, 2):
        methods = MAIN_KEYS[:5] if shot == 1 else MAIN_KEYS
        for ds, name in zip(DATASETS, NAMES):
            got = results.get(f"{ds}/{shot}", {}).get("methods", {})
            row = [title, shot, name]
            for key, label in zip(methods, MAIN_LABELS):
                item = got.get(key, {})
                acc = item.get("metrics", {}).get("accuracy")
                row.append(pct(acc))
                if acc is not None:
                    count += 1
                    add_flat(title, ds, shot, key, item, path)
                pair = item.get("paired_vs_parent")
                if pair:
                    parent = label.split("+Decision")[0]
                    record_pair(title, ds, shot, label, parent, pair, path)
            if shot == 1:
                row.extend(["—", "—"])
            MAIN_ROWS.append(row)
    return count


def generic_control():
    root = ART / "priority_1shot_ablation_20260920/qwen3_generic_direct_1_2shot_20260921"
    root_complete = verify_complete(root / "complete.json")
    assert len(root_complete["files"]) == 11
    obj = js(root / "live_results.json")
    main_path = ART / "qwen3_five_dataset_recalibrated_20260910/live_results.json"
    main_results = js(main_path).get("results", {})
    expected = 0
    for shot in (1, 2):
        parents = ["rices", "muier"] if shot == 1 else ["rices", "muier", "circles"]
        for ds, name in zip(DATASETS, NAMES):
            folder = root / f"{shot}shot" / ds
            complete = verify_complete(folder / "complete.json")
            audit = js(folder / "independent_audit.json")
            result = js(folder / "results.json")
            predictions = js(folder / "predictions.json")
            assert audit.get("status") == "passed" and audit.get("accuracy_recomputed") is True
            assert audit.get("dataset") == ds and audit.get("shots") == shot
            assert audit.get("methods") == len(parents) == len(result.get("methods", {})) == len(predictions)
            assert str((folder / "independent_audit.json").resolve()) in complete["files"]
            expected += len(parents)
            got = obj.get("results", {}).get(f"{ds}/{shot}", {}).get("methods", {})
            main_got = main_results.get(f"{ds}/{shot}", {}).get("methods", {})
            for parent in parents:
                parent_key = {"rices": MAIN_KEYS[1], "muier": MAIN_KEYS[2], "circles": MAIN_KEYS[5]}[parent]
                decision_key = {"rices": MAIN_KEYS[3], "muier": MAIN_KEYS[4], "circles": MAIN_KEYS[6]}[parent]
                parent_acc = main_got[parent_key]["metrics"]["accuracy"]
                decision_acc = main_got[decision_key]["metrics"]["accuracy"]
                method = parent + "+generic_visual_direct_sample"
                item = got[method]
                generic_acc = item["metrics"]["accuracy"]
                pair_parent = item["paired_vs_existing_parent"]
                pair_decision = item["paired_vs_existing_decision"]
                add_flat("Qwen3-VL-4B BF16", ds, shot, method, item, root / "live_results.json")
                record_pair("Qwen3-VL-4B BF16", ds, shot, parent.upper() + "+Generic", parent.upper(), pair_parent, root / "live_results.json")
                record_pair("Qwen3-VL-4B BF16", ds, shot, parent.upper() + "+Generic", parent.upper() + "+Decision", pair_decision, root / "live_results.json")
                ci = pair_decision.get("paired_bootstrap_95pct")
                GENERIC_ROWS.append([
                    shot, name, parent.upper(), pct(parent_acc), pct(decision_acc), pct(generic_acc),
                    f'{100 * pair_decision["accuracy_delta"]:+.3f}',
                    f'{pair_decision["rescue"]}/{pair_decision["harm"]}',
                    f'[{100 * ci[0]:+.3f}, {100 * ci[1]:+.3f}]' if ci else "—",
                    f'{pair_decision["exact_mcnemar_p"]:.6g}' if "exact_mcnemar_p" in pair_decision else "—"
                ])
    assert expected == 25 and obj.get("completed_method_results") == 25
    return expected


def gemma_main_table(queue):
    stage = next(s for s in queue["stages"] if s["id"] == "gemma_main")
    target = Path(stage["target_root"])
    roots = {k: Path(v) for k, v in stage.get("completed_result_roots", {}).items()}
    audited = 0
    audited_blocks = []
    for shot, method_count in ((1, 5), (2, 7)):
        for ds, name in zip(DATASETS, NAMES):
            block_key = f"{shot}shot/{ds}"
            folder = roots.get(block_key, target) / block_key
            rp, cp = folder / "results.json", folder / "complete.json"
            ap, pp = folder / "independent_audit.json", folder / "predictions.json"
            methods_expected = GEMMA_METHODS[:method_count]
            row = ["Gemma4 12B NF4", shot, name]
            if all(p.exists() for p in (rp, cp, ap, pp)):
                verify_complete(cp)
                result, audit, predictions = js(rp), js(ap), js(pp)
                plan_audit = js(folder / "independent_plan_audit.json")
                assert audit.get("status") == "passed" and plan_audit.get("status") == "passed"
                assert audit.get("results_sha256") == sha(rp)
                assert audit.get("complete_sha256") == sha(cp)
                assert audit.get("predictions_sha256") == sha(pp)
                assert audit.get("cache_provenance", {}).get("complete") is True
                assert plan_audit.get("test_labels_used") is False
                assert all(m in result.get("methods", {}) and m in audit.get("methods", {}) and m in predictions for m in methods_expected)
                for method in methods_expected:
                    saved = result["methods"][method]
                    checked = audit["methods"][method]
                    assert saved["metrics"] == checked["metrics"]
                    assert saved["invalid_exact_output_count"] == checked["invalid"]
                    if "+" in method:
                        assert saved["paired_vs_parent"] == checked["paired_vs_parent"]
                    row.append(pct(saved["metrics"].get("accuracy")))
                    add_flat("Gemma4 12B NF4", ds, shot, method, saved, rp)
                    pair = saved.get("paired_vs_parent")
                    if pair:
                        label = method.replace("+matched_decision", "+Decision")
                        record_pair("Gemma4 12B NF4", ds, shot, label, label.split("+Decision")[0], pair, rp)
                audited += method_count
                audited_blocks.append(block_key)
            else:
                row.extend(["待完成"] * method_count)
            if shot == 1:
                row.extend(["—", "—"])
            MAIN_ROWS.append(row)
    assert audited == stage.get("completed_method_results", audited), (
        f"queue Gemma count {stage.get('completed_method_results')} != independently matched {audited}"
    )
    pause_path = Path(stage.get("pause_receipt", "")) if stage.get("pause_receipt") else None
    snapshot = js(pause_path) if pause_path and pause_path.is_file() else {}
    return audited, stage, snapshot, audited_blocks


def compact_main_rows():
    """Show both shots and each parent/Decision pair in one cell."""
    by_key = {(row[0], row[1], row[2]): row for row in MAIN_ROWS}
    assert len(by_key) == 30
    def pair_line(shot, parent, decision):
        if parent == decision == "待完成":
            return f"{shot}: 待完成"
        return f"{shot}: {parent} → {decision}"

    rows = []
    for model in ("Qwen3-VL-4B BF16", "Qwen2.5-VL-7B NF4", "Gemma4 12B NF4"):
        for name in NAMES:
            one = by_key[(model, 1, name)]
            two = by_key[(model, 2, name)]
            rows.append([
                model, name,
                f"1: {one[3]}<br>2: {two[3]}",
                pair_line(1, one[4], one[6]) + "<br>" + pair_line(2, two[4], two[6]),
                pair_line(1, one[5], one[7]) + "<br>" + pair_line(2, two[5], two[7]),
                pair_line(2, two[8], two[9]),
            ])
    return rows


def compact_generic_rows():
    """Group the 25 audited comparisons into dataset-shot rows."""
    by_key = {(row[0], row[1], row[2]): row for row in GENERIC_ROWS}
    assert len(by_key) == 25
    rows = []
    for shot in (1, 2):
        for name in NAMES:
            line = [shot, name]
            for parent in ("RICES", "MUIER", "CIRCLES"):
                row = by_key.get((shot, name, parent))
                if row is None:
                    assert shot == 1 and parent == "CIRCLES"
                    line.append("—")
                    continue
                line.append(
                    f"P {row[3]} → D {row[4]} → G {row[5]}<br>"
                    f"G−D {row[6]} pp · R/H {row[7]}"
                )
            rows.append(line)
    return rows



def rotated_factor_class_control():
    """Accept only independently replayed, hash-bound completed control blocks."""
    root = ART / "priority_1shot_ablation_20260920/qwen3_rotated_factor_class_2shot_20261003"
    main = js(ART / "qwen3_five_dataset_recalibrated_20260910/live_results.json")["results"]
    generic = js(ART / "priority_1shot_ablation_20260920/qwen3_generic_direct_1_2shot_20260921/live_results.json")["results"]
    audited = 0
    for ds, name in zip(("dtd", "aircraft", "cub"), ("DTD", "Aircraft", "CUB")):
        folder = root / "2shot" / ds
        complete_path, result_path = folder / "complete.json", folder / "results.json"
        audit_path, plan_path = folder / "independent_audit.json", folder / "test_plan.json"
        parent = main[f"{ds}/2"]["methods"][MAIN_KEYS[1]]["metrics"]["accuracy"]
        decision = main[f"{ds}/2"]["methods"][MAIN_KEYS[3]]["metrics"]["accuracy"]
        generic_acc = generic[f"{ds}/2"]["methods"][GENERIC_KEYS[0]]["metrics"]["accuracy"]
        if not all(p.is_file() for p in (complete_path, result_path, audit_path, plan_path)):
            ROTATED_ROWS.append([name, pct(parent), pct(decision), pct(generic_acc), "待完成", "—", "—"])
            continue
        verify_complete(complete_path)
        result, audit = js(result_path), js(audit_path)
        assert audit.get("status") == "passed" and audit.get("dataset") == ds and audit.get("shots") == 2
        assert audit.get("complete_sha256") == sha(complete_path)
        assert audit.get("results_sha256") == sha(result_path)
        assert audit.get("plan_sha256") == sha(plan_path)
        assert audit.get("queries") == {"dtd": 1880, "aircraft": 3333, "cub": 4794}[ds]
        assert all(audit.get(k) is True for k in (
            "calibration_replayed", "plan_replayed", "raw_partner_and_parser_replayed", "paired_recomputed"))
        assert audit.get("test_used_for_tuning") is False and result.get("test_used_for_tuning") is False
        assert audit.get("method") == result.get("method") == "rices+rotated_decision_class_prototype"
        assert audit.get("accuracy") == result["metrics"]["accuracy"]
        assert audit.get("macro_accuracy") == result["metrics"]["macro_accuracy"]
        assert audit.get("invalid") == result["invalid_exact_output_count"]
        assert abs(result["metrics"]["accuracy"] - decision - result["paired_vs_matched_decision"]["accuracy_delta"]) < 1e-12
        audited += 1
        add_flat("Qwen3-VL-4B BF16", ds, 2, result["method"], result, result_path)
        for comparator, key in (("RICES", "paired_vs_parent"),
                                ("RICES+Decision", "paired_vs_matched_decision"),
                                ("RICES+Generic", "paired_vs_generic")):
            record_pair("Qwen3-VL-4B BF16", ds, 2, "RICES+RotatedDecision", comparator, result[key], result_path)
        pair = result["paired_vs_matched_decision"]
        ROTATED_ROWS.append([
            name, pct(parent), pct(decision), pct(generic_acc), pct(result["metrics"]["accuracy"]),
            f'{100 * pair["accuracy_delta"]:+.3f} pp · {pair["rescue"]}/{pair["harm"]}',
            link(audit_path, "独立审计")
        ])
    return audited


def visual_centroid_class_control():
    """Accept only independently replayed, hash-bound completed control blocks."""
    root = ART / "priority_1shot_ablation_20260920/qwen3_visual_centroid_class_gate_2shot_20261004"
    main = js(ART / "qwen3_five_dataset_recalibrated_20260910/live_results.json")["results"]
    generic = js(ART / "priority_1shot_ablation_20260920/qwen3_generic_direct_1_2shot_20260921/live_results.json")["results"]
    audited = 0
    resume_root = ART / "priority_1shot_ablation_20260920/qwen3_visual_centroid_class_gate_2shot_resume_20261004"
    for ds, name in zip(("dtd", "aircraft", "cub"), ("DTD", "Aircraft", "CUB")):
        selected_root = resume_root if ds == "cub" and resume_root.is_dir() else root
        folder = selected_root / "2shot" / ds
        complete_path, result_path = folder / "complete.json", folder / "results.json"
        audit_path, plan_path = folder / "independent_audit.json", folder / "test_plan.json"
        parent = main[f"{ds}/2"]["methods"][MAIN_KEYS[1]]["metrics"]["accuracy"]
        decision = main[f"{ds}/2"]["methods"][MAIN_KEYS[3]]["metrics"]["accuracy"]
        generic_acc = generic[f"{ds}/2"]["methods"][GENERIC_KEYS[0]]["metrics"]["accuracy"]
        if not all(p.is_file() for p in (complete_path, result_path, audit_path, plan_path)):
            VISUAL_CENTROID_ROWS.append([name, pct(parent), pct(decision), pct(generic_acc), "待完成", "—", "—"])
            continue
        if ds == "cub" and selected_root == resume_root:
            assert js(resume_root / "resume_preflight.json")["status"] == "passed"
            assert js(resume_root / "resume_independent_check.json")["status"] == "passed"
        verify_complete(complete_path)
        result, audit = js(result_path), js(audit_path)
        assert audit.get("status") == "passed" and audit.get("dataset") == ds and audit.get("shots") == 2
        assert audit.get("complete_sha256") == sha(complete_path)
        assert audit.get("results_sha256") == sha(result_path)
        assert audit.get("plan_sha256") == sha(plan_path)
        assert audit.get("queries") == {"dtd": 1880, "aircraft": 3333, "cub": 4794}[ds]
        assert all(audit.get(k) is True for k in (
            "calibration_replayed", "plan_replayed", "raw_partner_and_parser_replayed", "paired_recomputed"))
        assert audit.get("test_used_for_tuning") is False and result.get("test_used_for_tuning") is False
        assert audit.get("method") == result.get("method") == "rices+clip_centroid_class_gate"
        assert audit.get("accuracy") == result["metrics"]["accuracy"]
        assert audit.get("macro_accuracy") == result["metrics"]["macro_accuracy"]
        assert audit.get("invalid") == result["invalid_exact_output_count"]
        assert abs(result["metrics"]["accuracy"] - decision - result["paired_vs_matched_decision"]["accuracy_delta"]) < 1e-12
        audited += 1
        add_flat("Qwen3-VL-4B BF16", ds, 2, result["method"], result, result_path)
        for comparator, key in (("RICES", "paired_vs_parent"),
                                ("RICES+Decision", "paired_vs_matched_decision"),
                                ("RICES+Generic", "paired_vs_generic")):
            record_pair("Qwen3-VL-4B BF16", ds, 2, "RICES+CLIPCentroidGate", comparator, result[key], result_path)
        pair = result["paired_vs_matched_decision"]
        VISUAL_CENTROID_ROWS.append([
            name, pct(parent), pct(decision), pct(generic_acc), pct(result["metrics"]["accuracy"]),
            f'{100 * pair["accuracy_delta"]:+.3f} pp · {pair["rescue"]}/{pair["harm"]}',
            link(audit_path, "独立审计")
        ])
    return audited


def parent_class_gate_control():
    """Accept only full no-factor blocks with an independent hash-bound replay."""
    root = ART / "priority_1shot_ablation_20260920/qwen3_parent_class_gate_2shot_20261004"
    main = js(ART / "qwen3_five_dataset_recalibrated_20260910/live_results.json")["results"]
    audited = 0
    for ds, name in zip(("dtd", "aircraft", "cub"), ("DTD", "Aircraft", "CUB")):
        folder = root / "2shot" / ds
        complete_path, result_path = folder / "complete.json", folder / "results.json"
        audit_path, plan_path = folder / "independent_audit.json", folder / "test_plan.json"
        parent = main[f"{ds}/2"]["methods"][MAIN_KEYS[1]]["metrics"]["accuracy"]
        decision = main[f"{ds}/2"]["methods"][MAIN_KEYS[3]]["metrics"]["accuracy"]
        if not all(p.is_file() for p in (complete_path, result_path, audit_path, plan_path)):
            PARENT_GATE_ROWS.append([name, pct(parent), pct(decision), "待完成", "—", "—"])
            continue
        verify_complete(complete_path)
        result, audit = js(result_path), js(audit_path)
        assert audit.get("status") == "passed" and audit.get("dataset") == ds and audit.get("shots") == 2
        assert audit.get("complete_sha256") == sha(complete_path)
        assert audit.get("results_sha256") == sha(result_path)
        assert audit.get("plan_sha256") == sha(plan_path)
        assert audit.get("queries") == {"dtd": 1880, "aircraft": 3333, "cub": 4794}[ds]
        assert audit.get("validation_and_plan_replayed") is True
        assert audit.get("raw_partner_parser_and_provenance_replayed") is True
        assert audit.get("paired_recomputed") is True
        assert audit.get("test_used_for_tuning") is False and result.get("test_used_for_tuning") is False
        assert audit.get("method") == result.get("method") == "rices+parent_top1_class_gate_no_factors"
        assert audit.get("accuracy") == result["metrics"]["accuracy"]
        assert audit.get("macro_accuracy") == result["metrics"]["macro_accuracy"]
        assert audit.get("invalid") == result["invalid_exact_output_count"]
        assert abs(result["metrics"]["accuracy"] - decision - result["paired_vs_decision"]["accuracy_delta"]) < 1e-12
        audited += 1
        add_flat("Qwen3-VL-4B BF16", ds, 2, result["method"], result, result_path)
        record_pair("Qwen3-VL-4B BF16", ds, 2, "RICES+NoFactorClassGate",
                    "RICES", result["paired_vs_parent"], result_path)
        record_pair("Qwen3-VL-4B BF16", ds, 2, "RICES+NoFactorClassGate",
                    "RICES+Decision", result["paired_vs_decision"], result_path)
        pair = result["paired_vs_decision"]
        PARENT_GATE_ROWS.append([
            name, pct(parent), pct(decision), pct(result["metrics"]["accuracy"]),
            f'{100 * pair["accuracy_delta"]:+.3f} pp · {pair["rescue"]}/{pair["harm"]}',
            link(audit_path, "独立审计")
        ])
    return audited

def pending_ablation_section(queue):
    pending_states = {"pending", "queued", "implementation_pending", "implementation_in_progress", "running", "running_pending_independent_audit", "paused_by_user"}
    pending = []
    for stage in queue.get("stages", []):
        if stage.get("id") in {"gemma_main", "qwen3_main_three_seed_full_20261004"}:
            continue
        if stage.get("status") in pending_states:
            pending.append(stage)
    add("\n## 待跑消融")
    if pending:
        add("\n仅列仍获授权且尚未完成的消融；用户暂停的消融标为暂停，不代表正在推理。")
        table(["阶段", "状态", "模型", "shot", "数据集", "预计方法项", "说明"], [
            [s.get("id"), s.get("status"), ", ".join(s.get("target_models", [])),
             ", ".join(map(str, s.get("shots", []))), ", ".join(s.get("datasets", [])),
             s.get("expected_method_results", "—"), s.get("reason", s.get("current_note", "按队列合同执行"))]
            for s in pending
        ])
    else:
        add("\n当前没有待跑的已授权消融；Generic 与类别轮换负控的已审计结果见上文。取消或被替代的项目不列为待跑。")
    return pending


def seed_repeat_control():
    """Include only independently rechecked new-seed target blocks."""
    roots = (ART / "qwen3_main_three_seed_full_20261004",
             ART / "qwen3_main_three_seed_full_v3_isolated_resume_20261005")
    audited = 0
    for seed in (17, 73):
        for dataset, name in zip(DATASETS, NAMES):
            for shots in (1, 2):
                verified = []
                for root in roots:
                    candidate = root / f"seed{seed}" / f"{shots}shot" / dataset
                    candidate_receipt = candidate / "independent_recheck_20261005.json"
                    candidate_complete = candidate / "complete.json"
                    if not candidate_receipt.exists() or not candidate_complete.exists():
                        continue
                    candidate_audit = js(candidate_receipt)
                    if (candidate_audit.get("status") == "passed" and
                            candidate_audit.get("complete_sha256") == sha(candidate_complete)):
                        verified.append((candidate, candidate_receipt, candidate_complete, candidate_audit))
                if not verified:
                    continue
                block, receipt, complete, audit = verified[0]
                verify_complete(complete)
                assert audit["status"] == "passed"
                assert (audit["seed"], audit["dataset"], audit["shots"]) == (seed, dataset, shots)
                assert audit["complete_sha256"] == sha(complete)
                assert audit["fresh_target_outputs"] and audit["fixed_batch_partners_checked"]
                methods = js(block / "results.json")["methods"]
                assert len(methods) == (5 if shots == 1 else 7) == audit["methods"]
                assert set(methods) == set(audit["methods_summary"])
                for key, item in methods.items():
                    check = audit["methods_summary"][key]
                    assert abs(item["metrics"]["accuracy"] - check["accuracy"]) < 1e-12
                    assert abs(item["metrics"]["macro_accuracy"] - check["macro_accuracy"]) < 1e-12
                    assert item["invalid_exact_output_count"] == check["invalid"]
                audited += len(methods)
                SEED_REPEAT_ROWS.append([seed, name, shots,
                    ", ".join(f"{label} {pct(methods[key]['metrics']['accuracy'])}"
                              for key, label in zip(MAIN_KEYS, MAIN_LABELS) if key in methods),
                    link(receipt, "独立审计")])
    return audited


def main():
    queue_path = ART / "priority_1shot_ablation_20260920/task_queue.json"
    queue = js(queue_path)
    count_q3 = main_model("Qwen3-VL-4B BF16", "qwen3_five_dataset_recalibrated_20260910")
    count_q25 = main_model("Qwen2.5-VL-7B NF4", "qwen25_nf4_five_dataset_20260913")
    assert count_q3 == 60 and count_q25 == 60, f"expected 120 Qwen main methods; got {count_q3}+{count_q25}"
    generic_count = generic_control()
    assert generic_count == 25
    audited_gemma, gemma_stage, snapshot, audited_blocks = gemma_main_table(queue)
    assert len(MAIN_ROWS) == 30 and len(GENERIC_ROWS) == 25
    assert len(FLAT) == 120 + generic_count + audited_gemma

    add("# Decision-Guided Retrieval 实验结果")
    run_status = "用户暂停" if gemma_stage.get("status") == "paused_by_user" else ("正式评估进行中" if gemma_stage.get("status") == "target_running" else str(gemma_stage.get("status")))
    add("\n更新：" + datetime.now().astimezone().isoformat(timespec="seconds") + "。汇总主试验、Generic 对照及已审计的类别轮换负控与已审计视觉类质心对照；Gemma " + run_status + "。本页仅整理和核验已有回执，生成脚本不运行推理。")
    add("\n数字为目标模型 accuracy（%）。每行是一组模型 × 数据集；格内 1、2 分别表示 1-shot、2-shot，箭头左侧是 parent，右侧是对应 Decision。‘待完成’表示尚无完整且独立审计通过的结果，‘—’表示该 shot 不适用。Decision 在完整训练池直接检索；不使用 test 调参。")
    add("\n## 主试验总表")
    add()
    table(["模型", "数据集", "Random", "RICES → Decision", "MUIER → Decision", "CIRCLES → Decision"], compact_main_rows())
    add(f"Qwen3 与 Qwen2.5 各 60/60 项已完成；Gemma 已独立审计 {audited_gemma}/60 项，当前状态：{run_status}。" +
        " Gemma 1-shot 每块五方法，2-shot 每块七方法；其 CIRCLES 使用本模型描述。")
    if gemma_stage.get("status") == "paused_by_user" and snapshot:
        add(f"\nGemma 暂停于 {snapshot.get('dataset')} {snapshot.get('shots')}-shot，缓存 {snapshot.get('durable_cached_contexts')}/{snapshot.get('expected_contexts')}；该部分没有 complete 与 independent_audit，不计入上表。已审计块：" + ", ".join(audited_blocks) + "。")
    add("\n主表来源：" + link(ART / "qwen3_five_dataset_recalibrated_20260910/live_results.json", "Qwen3") + "、" +
        link(ART / "qwen25_nf4_five_dataset_20260913/live_results.json", "Qwen2.5") + "；Gemma 按" +
        link(queue_path, "队列中的独立审计根") + "逐块核验。")

    seed_repeat_audited = seed_repeat_control()
    add("\n## Qwen3 三 seed 完整目标重跑（进行中）")
    add("\n原 seed42 见主表；新 seed17、73 的五数据集 1/2-shot 全部方法重新执行目标推理。因素库与视觉证据固定，Random 示例与 validation 五折随 seed 变化，Decision 逐 seed 独立校准。下表只收 complete 哈希和单独独立复核回执匹配的完整块；数值为 accuracy（%）。")
    add()
    table(["Seed", "数据集", "Shot", "方法准确率", "回执"], SEED_REPEAT_ROWS)
    add(f"新增方法项已独立审计 {seed_repeat_audited}/120；部分推理缓存和运行器自身的完成回执不计入。")

    add("\n## Generic 关键消融（Qwen3，已完成）")
    add("\nGeneric 在完整训练池逐样本直接检索，并按 dataset × shot × parent 独立校准。每行是一组数据集 × shot；格内 P / D / G 分别为 parent / Decision / Generic 的 accuracy（%）。G−D 与 R/H（rescue/harm）均为 **Generic 相对 Decision** 的配对比较；95% CI 和双侧 exact 原始 p 见下方配对统计 CSV。1-shot 不运行 CIRCLES。")
    add()
    table(["Shot", "数据集", "RICES", "MUIER", "CIRCLES"], compact_generic_rows())
    add("25/25 项完成，十块 complete 哈希闭包与 accuracy 独立审计通过；" +
        link(ART / "priority_1shot_ablation_20260920/qwen3_generic_direct_1_2shot_20260921/live_results.json", "原始结果") + "。")

    rotated_audited = rotated_factor_class_control()
    add("\n## 类别轮换负控（Qwen3，2-shot RICES 分支）")
    add("\n将 train-only 因子原型的类别对应关系循环轮换一位，并针对这一变体独立重做 train/validation 校准；其余模型、证据、完整训练池检索和目标推理合同不变。P / D / G / 轮换分别为 RICES、Decision、Generic 与该负控的 accuracy（%）。差值、R/H 均为轮换相对 Decision；95% CI 与双侧 exact 原始 p 见配对统计 CSV。")
    add()
    table(["数据集", "P", "D", "G", "轮换", "轮换−D · R/H", "回执"], ROTATED_ROWS)
    add(f"已独立审计 {rotated_audited}/3 项；每项仅在 complete 哈希闭包、校准/计划/原始输出独立重放及配对指标核验通过后入账。CUB 的 validation 校准选择 alpha=1 parent-only 端点，该块不能单独解释为类别错配造成的性能变化。")

    visual_audited = visual_centroid_class_control()
    add("\n## 视觉类质心门控对照（Qwen3，2-shot RICES 分支）")
    add("\n以冻结 CLIP 训练类质心替代目标模型因素证据，其余类别门控、完整训练池类内检索、validation 校准与固定 batch2 目标推理合同不变。P / D / G / 视觉分别为 RICES、Decision、Generic 与该对照的 accuracy（%）。差值、R/H 均为视觉相对 Decision；95% CI 与双侧 exact 原始 p 见配对统计 CSV。")
    add()
    table(["数据集", "P", "D", "G", "视觉", "视觉−D · R/H", "回执"], VISUAL_CENTROID_ROWS)
    add(f"已独立审计 {visual_audited}/3 项；仅接纳 complete 哈希闭包、校准/计划/固定批伙伴/原始输出及配对指标独立重放通过的结果。")

    parent_gate_audited = parent_class_gate_control()
    add("\n## 去因素同类别门控消融（Qwen3，2-shot RICES）")
    add("\n仅用 RICES 全训练池 top-1 的类别选类，再在该类的完整训练池取 RICES top-2；无因素证据、无可学习参数。P / D / 去因素分别为 RICES、Decision 与本消融 accuracy（%）；差值及救回/误伤为去因素相对 Decision。完整 paired CI 与原始 exact p 见配对统计 CSV。")
    add()
    table(["数据集", "P", "D", "去因素", "去因素−D · R/H", "回执"], PARENT_GATE_ROWS)
    add(f"已独立审计 {parent_gate_audited}/3 项；只接纳完整 test、冻结计划、固定 batch raw 来源、严格解析及配对统计独立重放通过的结果。")
    pending = pending_ablation_section(queue)
    add("\n## 核验与逐项数据")
    add("\n主表的各 Decision-vs-parent、Generic 及类别轮换负控、视觉类质心门控比较，以及 Gemma 已完成块的配对 rescue/harm、paired bootstrap 95% CI 与未校正双侧 exact McNemar 原始 p，均在" +
        link(OUT.with_suffix(".paired.csv"), "配对统计 CSV") + "。逐方法 accuracy、macro 与 invalid 计数在" +
        link(OUT.with_suffix(".csv"), "方法结果 CSV") + "；" +
        link(OUT.with_suffix(".sources.json"), "来源 SHA256 清单") + "可追溯输入。")
    add("\nGemma 仅接受 complete、results、predictions、计划、缓存 provenance 和 independent_audit 哈希匹配的完整块；partial cache 不计完成。VICL、Prompt-SelF、MMICES 辅助协议与历史报告仍保留在 artifacts，但不展开在本总账；没有授权 Gemma VICL 或 Prompt-SelF。")
    add("\n更新脚本：" + link(ROOT / "scripts/build_experiment_results_ledger.py", "ledger generator") +
        "；研究主线：" + link(ROOT / "docs/iclr_decision_guided_mainline_zh.md", "mainline") +
        "；Gemma 实施合同：" + link(CLEAN / "docs/gemma12_main_1_2shot_v3_20260926_zh.md", "协议") + "。")
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    csv_path = OUT.with_suffix(".csv")
    if FLAT:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(FLAT[0]))
            writer.writeheader()
            writer.writerows(FLAT)
    if PAIRED:
        with OUT.with_suffix(".paired.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(PAIRED[0]))
            writer.writeheader()
            writer.writerows(PAIRED)
    OUT.with_suffix(".sources.json").write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(),
        "structured_result_rows": len(FLAT),
        "paired_result_rows": len(PAIRED),
        "main_qwen_method_results": count_q3 + count_q25,
        "generic_control_method_results": generic_count,
        "gemma_independently_audited_method_results": audited_gemma,
        "rotated_factor_class_audited_method_results": rotated_audited,
        "visual_centroid_class_audited_method_results": visual_audited,
        "parent_class_gate_audited_method_results": parent_gate_audited,
        "qwen3_seed_repeat_independently_audited_method_results": seed_repeat_audited,
        "pending_ablation_stages": [s.get("id") for s in pending],
        "sources_sha256": SOURCES
    }, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"document": str(OUT), "lines": len(LINES), "structured_rows": len(FLAT),
        "paired_rows": len(PAIRED), "qwen_main_methods": count_q3 + count_q25,
        "generic_control_methods": generic_count, "gemma_audited_methods": audited_gemma,
        "rotated_audited_methods": rotated_audited,
        "visual_centroid_audited_methods": visual_audited,
        "parent_class_gate_audited_methods": parent_gate_audited,
        "pending_ablations": len(pending), "source_files": len(SOURCES)}, ensure_ascii=True))


if __name__ == "__main__":
    main()

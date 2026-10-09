"""Read-only audit of completed predictions; writes only reporting artifacts."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/five_dataset_full_suite_20260905"
DATASETS = ("cub", "pets", "dtd", "dogs", "aircraft")
RAW = ("random", "rices", "muier", "equal_alpha", "nnls_raw", "support_equal",
       "rices_atom_residual_raw", "muier_atom_equal_alpha_raw", "muier_atom_nnls_raw",
       "atom_label_equal_alpha_raw", "atom_label_residual_raw", "atom_label_gated_raw")


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    output, previous = [0.0] * len(values), 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, values[index] * (len(values) - rank)))
        output[index] = previous
    return output


def fmt_p(value):
    return "<1e-300 (underflow)" if value == 0 else f"{value:.4g}"


def main():
    results, hashes, records = {}, {}, []
    for dataset in DATASETS:
        path = OUT / dataset / "qwen3_results.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["status"] == "complete" and not result["pending_methods"]
        results[dataset] = result
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = (ROOT / "artifacts/cub_full_current_atoms_20260904/cub.test.jsonl"
                    if dataset == "cub" else ROOT /
                    f"artifacts/four_dataset_atom_label_20260905/{dataset}/{dataset}.test.jsonl")
        truth_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        # Manifest schema uses sample_id.
        truth = {row["sample_id"]: row["label"] for row in truth_rows}
        predictions_path = OUT / dataset / "qwen3_predictions.json"
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        hashes[str(predictions_path.relative_to(ROOT))] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
        hashes[str(manifest.relative_to(ROOT))] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        assert len(truth) == result["query_count"] == len(truth_rows)
        correctness = {}
        for method, metric in result["methods"].items():
            assert set(predictions[method]) == set(truth)
            correctness[method] = np.array([predictions[method][q] == truth[q] for q in truth])
            assert abs(correctness[method].mean() - metric["accuracy"]) < 1e-12
            per_class = {}
            for q in truth:
                per_class.setdefault(truth[q], []).append(predictions[method][q] == truth[q])
            macro = np.mean([np.mean(v) for v in per_class.values()])
            assert abs(macro - metric["macro_accuracy"]) < 1e-12
        for key, paired in result["paired"].items():
            method, baseline = key.rsplit("_vs_", 1)
            current, base = correctness[method], correctness[baseline]
            rescue, harm = int((current & ~base).sum()), int((~current & base).sum())
            assert (rescue, harm) == (paired["rescue"], paired["harm"])
            p = float(binomtest(rescue, rescue + harm, 0.5).pvalue) if rescue + harm else 1.0
            assert abs(p - paired["exact_mcnemar_p"]) < 1e-12
            delta = (rescue - harm) / len(truth)
            assert abs(delta - paired["accuracy_delta"]) < 1e-12
            records.append(dict(dataset=dataset, method=method, baseline=baseline, **paired))
    methods = list(results["cub"]["methods"])
    assert all(set(r["methods"]) == set(methods) for r in results.values())
    for method in methods:
        for baseline in ("rices", "muier"):
            family = [r for r in records if r["method"] == method and r["baseline"] == baseline]
            if not family:
                continue
            assert len(family) == 5
            for record, adjusted in zip(family, holm([r["exact_mcnemar_p"] for r in family])):
                record["holm_within_method_five_datasets"] = adjusted
    # Broad sensitivity family: 10 non-baseline raw variants x 2 baselines x 5 tasks,
    # plus MUIER vs RICES x 5. Excludes deployment/MMICES aliases and reverse baseline pair.
    broad = [r for r in records if r["method"] in RAW and
             not (r["method"] == "rices" and r["baseline"] == "muier")]
    assert len(broad) == 105, len(broad)
    for record, adjusted in zip(broad, holm([r["exact_mcnemar_p"] for r in broad])):
        record["holm_broad_105_comparisons"] = adjusted
    payload = dict(
        scope="Completed Qwen3-VL-4B-Instruct clean-room full test, 1-shot, batch=1",
        exploratory=True, verified_against_complete_predictions=True,
        correction_note="Post-hoc reporting only; not method selection. Holm-5 covers datasets within each fixed comparison, not method search. Holm-105 is a broader sensitivity analysis, not correction for all historical research.",
        source_sha256=hashes, datasets={d: results[d]["query_count"] for d in DATASETS},
        metrics={d: results[d]["methods"] for d in DATASETS}, paired=records)
    (OUT / "significance_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 五数据集完整表现与配对显著性", "",
             "Qwen3-VL-4B-Instruct，完整 test，1-shot，batch=1；不含正在运行的迭代 atom 或后续新 Gemma。", "",
             "已从逐 query 预测及 test manifest 复算 accuracy、macro accuracy、全部 rescue/harm 和双侧 exact McNemar p；所有值与原结果一致。此分析不更改任何检索参数或测试预测。", "",
             "数据量：" + "；".join(f"{d}={results[d]['query_count']}" for d in DATASETS) + "。", "",
             "## 统计口径", "",
             "- p 使用双侧 exact McNemar；bootstrap 区间沿用冻结评估程序的 10,000 次逐样本配对 percentile 95% CI。区间未做多重校正，也不涵盖重新发现 atoms/训练的种子方差。",
             "- Holm-5：每一个固定方法对固定基线，跨五个数据集构成一个检验族。这不能校正同时尝试多种方法带来的选择风险。",
             "- Holm-105：12 个非部署方法（含 Random/RICES/MUIER，MMICES 与 RICES 合并）中，10 个非基线各对两基线，加 MUIER 对 RICES，乘五数据集；去除基线反向重复，共 105 次比较。此为更保守的事后敏感性分析，也不覆盖全部历史搜索。",
             "- 所有校正只用于报告，不用于 test 上选择方法。显著性阈值 0.05；p 小但 delta 为负是显著变差。",
             "- 这批数据属于探索性证据；CUB equal 层是后来授权追加，不能声称预注册确认性。不能将五项 test 最优拼接为一个新部署算法。", ""]
    for metric, title in (("accuracy", "目标模型 Accuracy (%)"), ("macro_accuracy", "目标模型 Macro accuracy (%)")):
        lines += ["## " + title, "", "| method | CUB | Pets | DTD | Dogs | Aircraft | 五数据集等权平均 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for method in methods:
            values = [100 * results[d]["methods"][method][metric] for d in DATASETS]
            lines.append("| " + method + " | " + " | ".join(f"{v:.3f}" for v in values) + f" | {np.mean(values):.3f} |")
        lines.append("")
    lines += ["## 方法与别名", "",
              "- MUIER 为 classification-adapted 实现，不是未经调整的论文原版。",
              "- MMICES 为 question-only 分支，固定问题全部同分、并列保留 RICES 顺序，因此本次预测完全相同；不能当成第二份独立获胜证据。",
              "- raw 是方法原始输出；deployed 是预定 validation/OOF 规则选择或回退后的输出。",
              "- atom_label_deployed 含 CUB 早期 gated 规则和其余 equal 规则的历史差异；不冒充统一新部署方法。",
              "- MLP raw 五项按失败规则跳过；mlp_deployed 是回退结果，不是独立 MLP 成果。CIRCLES 按用户要求未运行。",
              "- 类别 equal 的 atoms 内部等权，但与基线的整体 alpha 在 validation 独立学习；不是固定 0.5 融合。",
              "- bootstrap CI 与 exact McNemar 是不同统计程序，DTD 类别 equal 的未校正 CI 略高于零，但 McNemar p=0.05590；按本文 p 检验口径仍不显著。", "",
              "## 所有配对统计", "",
              "Δ 和 CI 单位均为百分点；救回=基线错、本方法对，误伤=基线对、本方法错。", "",
              "| 方法 | 基线 | 数据集 | Δpp | 救回/误伤 | bootstrap 95% CI | p | Holm-5 p | Holm-105 p |",
              "|---|---|---|---:|---:|---|---:|---:|---:|"]
    for r in records:
        ci = [100 * v for v in r["paired_bootstrap_95pct"]]
        broad_p = fmt_p(r["holm_broad_105_comparisons"]) if "holm_broad_105_comparisons" in r else "不入该族"
        lines.append(f"| {r['method']} | {r['baseline']} | {r['dataset']} | {100*r['accuracy_delta']:+.3f} | {r['rescue']}/{r['harm']} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {fmt_p(r['exact_mcnemar_p'])} | {fmt_p(r['holm_within_method_five_datasets'])} | {broad_p} |")
    (OUT / "significance_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verified_datasets": payload["datasets"], "paired_comparisons": len(records),
                      "broad_family": len(broad),
                      "key_results": [r for r in records if r["method"] == "atom_label_equal_alpha_raw" and r["baseline"] == "muier"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

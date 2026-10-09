"""Compare ORIGINAL atoms using full train/validation evidence; no target/test access."""
from __future__ import annotations
import argparse
import dataclasses
import gc
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.stats import binomtest, spearmanr, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fgclip2_evidence import FGClip2Encoder, MODEL, PROJECT, REVISION, WEIGHT_HASH
from dgdr.evidence import (EmbeddingStore, EvidenceStore, build_evidence_store,
    fit_train_evidence_calibration, apply_evidence_calibration, _derive_group_arrays)
from dgdr.io import load_atom_bank, load_manifest, read_json, read_jsonl, write_json
from dgdr.hashing import file_sha256, stable_hash
from dgdr.audit import distribution_audit
from audit_cub_official_attributes import load_positive_attributes, family_mapper

OUT = PROJECT / "artifacts/fgclip2_evidence_comparison_20260906"
PROTOCOL = PROJECT / "docs/fgclip2_evidence_protocol_20260906_zh.md"
DATASETS = ("cub", "pets", "dtd", "dogs", "aircraft")

def locations(ds):
    data = PROJECT / "artifacts" / ("cub_full_current_atoms_20260904" if ds == "cub" else "four_dataset_atom_label_20260905/" + ds)
    bank = PROJECT / "artifacts/cub_siglip_flair_pilot/atom_bank.final.json" if ds == "cub" else data / "atom_bank.final.json"
    return data, bank, "siglip." if ds == "cub" else ""

def held():
    if any((OUT / n).exists() for n in ("PAUSED_BY_USER.json", "DIAGNOSTIC_HOLD.json")):
        raise RuntimeError("new comparison is on hold")

def event(stage, **values):
    row = dict(stage=stage, updated_at=time.time(), pid=os.getpid(), **values)
    write_json(OUT / "status.json", row)
    print(json.dumps(row), flush=True)

def freeze(path, files):
    write_json(path, dict(files={str(p): file_sha256(p) for p in files}))

def verify(path):
    for p, digest in read_json(path)["files"].items():
        if file_sha256(p) != digest:
            raise ValueError("frozen file drift: " + p)
        if Path(p).suffix == ".json" and "files" in read_json(p):
            verify(p)

def raw_store(store, bank):
    p, s, o, a = _derive_group_arrays(raw_logits=store.raw_logits, bank=bank,
        state_group_array=store.state_group_index, state_logit_bias=np.zeros(len(store.state_ids)))
    return dataclasses.replace(store, state_probabilities=p.astype(np.float32), signed_margins=s.astype(np.float32),
        ordinal_scores=o.astype(np.float32), applicability_scores=a.astype(np.float32), calibration_hash="identity",
        transform=store.transform.replace("__train-state-mean-debias__", "__uncalibrated__").replace("log-margin-v2", "log-margin-v1"))

def check(store, bank, samples):
    if store.bank_hash != bank.bank_hash or store.sample_ids != tuple(s.sample_id for s in samples):
        raise ValueError("evidence bank/sample alignment mismatch")
    if store.state_ids != tuple(f"{g.group_id}/{s.state_id}" for g in bank.groups for s in g.states):
        raise ValueError("state order mismatch")
    store.validate()

def preflight():
    from transformers import AutoTokenizer
    from dgdr.evidence import DEFAULT_TEXT_TEMPLATE
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    report = {}
    for ds in DATASETS:
        data, bp, prefix = locations(ds)
        bank = load_atom_bank(bp)
        samples = {s: load_manifest(data / f"{ds}.{s}.jsonl", expected_split=s) for s in ("train", "validation")}
        stores = {s: EvidenceStore.load(data / f"{prefix}{s}.evidence") for s in samples}
        for s in samples:
            check(stores[s], bank, samples[s])
            if stores[s].text_template != DEFAULT_TEXT_TEMPLATE:
                raise ValueError("original evidence text template differs from comparison")
        if set(stores["train"].sample_ids) & set(stores["validation"].sample_ids):
            raise ValueError("overlapping train/validation samples")
        raw = raw_store(stores["train"], bank)
        calibration = fit_train_evidence_calibration(samples=samples["train"], bank=bank, uncalibrated_store=raw)
        if any(stores[s].calibration_hash != calibration.calibration_hash for s in samples):
            raise ValueError("SigLIP independent calibration parity failed: " + ds)
        rebuilt = apply_evidence_calibration(store=raw, bank=bank, calibration=calibration)
        np.testing.assert_array_equal(rebuilt.state_probabilities, stores["train"].state_probabilities)
        texts = [DEFAULT_TEXT_TEMPLATE.format(description=s.visual_description) for g in bank.groups for s in g.states]
        lengths = [len(x) for x in tokenizer(texts, truncation=False)["input_ids"]]
        if max(lengths) > 64:
            raise ValueError("short text limit exceeded: " + ds)
        report[ds] = dict(split_counts={s:len(v) for s,v in samples.items()}, groups=len(bank.groups),
            states=len(texts), max_text_tokens=max(lengths), independent_siglip_calibration_exact=True,
            bank_hash=bank.bank_hash, test_used=False)
    write_json(OUT / "preflight.json", report)
    print(json.dumps(report), flush=True)
    return report

def prepare(ds, encoder):
    held()
    data, bp, prefix = locations(ds)
    folder = OUT / ds
    bank = load_atom_bank(bp)
    samples = {s: load_manifest(data / f"{ds}.{s}.jsonl", expected_split=s) for s in ("train", "validation")}
    sig = {s: EvidenceStore.load(data / f"{prefix}{s}.evidence") for s in samples}
    for s in samples:
        check(sig[s], bank, samples[s])
    if set(sig["train"].sample_ids) & set(sig["validation"].sample_ids):
        raise ValueError("train/validation overlap")
    files = [PROTOCOL, Path(__file__), Path(__file__).with_name("fgclip2_evidence.py"),
        Path(__file__).with_name("audit_cub_official_attributes.py"), bp]
    files += list((PROJECT / "src/dgdr").glob("*.py"))
    files += [p for p in MODEL.iterdir() if p.suffix in (".py", ".json", ".model", ".safetensors")]
    for s in samples:
        files += [data / f"{ds}.{s}.jsonl"] + [data / f"{prefix}{s}.evidence.{e}" for e in ("json", "npz")]
    contract = dict(files={str(p): file_sha256(p) for p in files}, bank_hash=bank.bank_hash,
        split_counts={s: len(v) for s, v in samples.items()}, evidence_encoder=encoder.model_revision,
        test_used=False, openai_bank_used=False, scoring="global image-text; not spatial grounding")
    contract_path = folder / "input_contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("comparison input drift")
    write_json(contract_path, contract)
    marker = folder / "evidence_complete.json"
    if marker.exists():
        verify(marker)
        return samples, bank, sig, {s: EvidenceStore.load(folder / f"fgclip2.{s}.evidence") for s in samples}
    fg = {}
    for split in ("train", "validation"):
        chunks = []
        for start in range(0, len(samples[split]), 256):
            held()
            rows = samples[split][start:start + 256]
            cp = folder / "embedding_chunks" / f"{split}.{start:06d}"
            done = Path(str(cp) + ".freeze.json")
            if done.exists():
                verify(done)
                emb = EmbeddingStore.load(cp)
            else:
                for sample in rows:
                    if not sample.image_sha256 or file_sha256(sample.image_path) != sample.image_sha256:
                        raise ValueError("image content mismatch: " + sample.sample_id)
                values = encoder.encode_images([s.image_path for s in rows])
                emb = EmbeddingStore(tuple(s.sample_id for s in rows), values, encoder.model_id, encoder.model_revision)
                emb.save(cp)
                freeze(done, [Path(str(cp) + "." + ext) for ext in ("json", "npz")])
            if emb.sample_ids != tuple(s.sample_id for s in rows) or emb.model_revision != encoder.model_revision:
                raise ValueError("embedding chunk contract mismatch")
            chunks.append(emb)
            event("evidence", dataset=ds, split=split, complete=start + len(rows), total=len(samples[split]))
        embeddings = EmbeddingStore(tuple(s.sample_id for s in samples[split]),
            np.concatenate([e.embeddings for e in chunks]), encoder.model_id, encoder.model_revision)
        embeddings.save(folder / f"fgclip2.{split}.embeddings")
        raw = build_evidence_store(samples=samples[split], image_embeddings=embeddings, bank=bank, encoder=encoder)
        if split == "train":
            bias = fit_train_evidence_calibration(samples=samples[split], bank=bank, uncalibrated_store=raw)
            bias.save(folder / "fgclip2.train_calibration.json")
            # Independently refit even the unchanged SigLIP calibration and require exact parity.
            sr = raw_store(sig[split], bank)
            sb = fit_train_evidence_calibration(samples=samples[split], bank=bank, uncalibrated_store=sr)
            sb.save(folder / "siglip.train_calibration.json")
            if sb.calibration_hash != sig[split].calibration_hash:
                raise ValueError("original SigLIP train calibration parity failed")
        fg[split] = apply_evidence_calibration(store=raw, bank=bank, calibration=bias)
        check(fg[split], bank, samples[split])
        fg[split].save(folder / f"fgclip2.{split}.evidence")
    freeze(marker, [contract_path, folder / "fgclip2.train_calibration.json", folder / "siglip.train_calibration.json"] +
        [folder / f"fgclip2.{s}.evidence.{ext}" for s in samples for ext in ("json", "npz")])
    return samples, bank, sig, fg

def compare_correct(left, right):
    left, right = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    rescue, harm = int((~left & right).sum()), int((left & ~right).sum())
    delta = right.astype(float) - left.astype(float)
    rng = np.random.default_rng(42)
    # Multinomial counts are equivalent to paired row bootstrap of {-1,0,1}.
    counts = rng.multinomial(len(delta), [harm / len(delta), 1 - (harm + rescue) / len(delta), rescue / len(delta)], size=10000)
    ci = np.quantile((counts[:, 2] - counts[:, 0]) / len(delta), [.025, .975])
    return dict(n=len(delta), siglip_accuracy=float(left.mean()), fgclip2_accuracy=float(right.mean()),
        rescue=rescue, harm=harm, delta=float(delta.mean()), ci95=ci.tolist(),
        exact_mcnemar_p=float(binomtest(rescue, rescue + harm, .5).pvalue) if rescue + harm else 1.)

def binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if not npos or not nneg:
        raise ValueError("AUROC requires both positive and negative examples")
    ranks = rankdata(np.asarray(scores), method="average")
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def state_metrics(store, j, indices, truth, ordered=False):
    p = store.group_probability(j)[indices]
    margins = store.group_signed_margin(j)[indices]
    truth = np.asarray(truth)
    correct = p.argmax(1) == truth
    per_state, aucs, balanced_sign = {}, [], []
    for k in range(p.shape[1]):
        y = truth == k
        if y.any() and (~y).any():
            auc = binary_auc(y, margins[:, k]); aucs.append(auc)
            ba = .5 * ((margins[y, k] > 0).mean() + (margins[~y, k] <= 0).mean())
            balanced_sign.append(float(ba))
            per_state[str(k)] = dict(n=int(y.sum()), recall=float(correct[y].mean()), auroc=auc, sign_balanced_accuracy=float(ba))
    rho = None
    if ordered and len(set(truth.tolist())) > 1:
        rho = float(spearmanr(truth, store.ordinal_scores[indices, j]).statistic)
        if not np.isfinite(rho): rho = None
    metrics = dict(n=len(truth), accuracy=float(correct.mean()),
        macro_state_recall=float(np.mean([correct[truth == k].mean() for k in np.unique(truth)])),
        true_state_positive_rate=float((margins[np.arange(len(truth)), truth] > 0).mean()),
        macro_auroc=float(np.mean(aucs)) if aucs else None,
        macro_sign_balanced_accuracy=float(np.mean(balanced_sign)) if balanced_sign else None,
        ordinal_spearman=rho, per_state=per_state)
    return metrics, correct

def cub_audit(samples, bank, sig, fg):
    attrs_path = PROJECT.parent / "data/cub_200_2011_raw/CUB_200_2011/attributes/image_attribute_labels.txt"
    attrs = load_positive_attributes(attrs_path, 3)
    specs = {
        "beak--length--batch-0000--58068026--00": ("bill_length", range(150, 153), {152:"short-0",150:"medium-1",151:"long-2"}),
        "body--coloration--batch-0000--58068026--02": ("primary_color", range(249, 264), {260:"black-0",255:"yellow-1",250:"brown-2"}),
        "beak--beak-color--batch-0000--a72c67dd--02": ("bill_color_gray_proxy", range(279, 294), {290:"black-0",284:"dark-gray-1",289:"orange-2"})}
    report = dict(official_attributes_sha256=file_sha256(attrs_path), official={}, human={})
    for split in samples:
        report["official"][split] = {}
        for j, group in enumerate(bank.groups):
            if group.group_id not in specs: continue
            name, family, mapping = specs[group.group_id]
            mapper = family_mapper(family, mapping)
            lookup = {s.state_id:k for k,s in enumerate(group.states)}
            indices, truth, labeled = [], [], 0
            for i, sample in enumerate(samples[split]):
                valid, state = mapper(attrs.get(int(sample.sample_id.removeprefix("cub-")), set()))
                labeled += int(valid)
                if valid and state in lookup:
                    indices.append(i); truth.append(lookup[state])
            item = dict(group_id=group.group_id, labeled=labeled, covered=len(indices),
                state_space_coverage=len(indices)/labeled if labeled else None, variants={})
            if indices:
                for calibration in ("train_debiased", "native_raw"):
                    ss = sig[split] if calibration == "train_debiased" else raw_store(sig[split], bank)
                    ff = fg[split] if calibration == "train_debiased" else raw_store(fg[split], bank)
                    sm, sc = state_metrics(ss, j, indices, truth, name == "bill_length")
                    fm, fc = state_metrics(ff, j, indices, truth, name == "bill_length")
                    item["variants"][calibration] = dict(siglip=sm, fgclip2=fm, paired=compare_correct(sc, fc))
            report["official"][split][name] = item
    annotations_path = PROJECT / "artifacts/cub_siglip_flair_pilot/human_audit/blind_annotations_20260904.jsonl"
    annotations = read_jsonl(annotations_path)
    report["human"]["source_sha256"] = file_sha256(annotations_path)
    report["human"]["selection_bias"] = "Historical assistant visual review, SigLIP extreme-enriched; not independent expert annotations or random population coverage"
    all_left, all_right, groups, missing = [], [], {}, []
    for j, group in enumerate(bank.groups):
        left, right = [], []
        for row in annotations:
            if row["group_id"] != group.group_id or not row["applicable"]: continue
            split = next((s for s in samples if row["sample_id"] in sig[s].sample_ids), None)
            if split is None:
                missing.append(row["sample_id"]); continue
            i = sig[split].sample_ids.index(row["sample_id"])
            expected = [s.state_id for s in group.states].index(row["observed_state_id"])
            left.append(sig[split].group_probability(j)[i].argmax() == expected)
            right.append(fg[split].group_probability(j)[i].argmax() == expected)
        if left:
            groups[group.group_id] = compare_correct(left, right)
            all_left.extend(left); all_right.extend(right)
    report["human"].update(groups=groups, missing=missing,
        pooled=compare_correct(all_left, all_right) if all_left else None,
        note="Pooled pairs repeat images/groups; uncertainty does not account for clustering")
    write_json(OUT / "cub/reliability_audit.json", report)
    return report

def diagnostics(ds, samples, bank, sig, fg):
    report = dict(dataset=ds, groups=len(bank.groups), states=len(sig["train"].state_ids), splits={},
        claim="distribution and agreement alone do not verify image evidence", test_used=False)
    for split in samples:
        per_group = {}
        for j, g in enumerate(bank.groups):
            sp, fp = sig[split].group_probability(j), fg[split].group_probability(j)
            sm, fm = sig[split].group_signed_margin(j), fg[split].group_signed_margin(j)
            per_group[g.group_id] = dict(top1_agreement=float((sp.argmax(1) == fp.argmax(1)).mean()),
                state_margin_spearman=[float(spearmanr(sm[:,k], fm[:,k]).statistic) if np.std(sm[:,k])>0 and np.std(fm[:,k])>0 else None for k in range(sm.shape[1])])
        report["splits"][split] = dict(n=len(samples[split]), groups=per_group,
            siglip_distribution=distribution_audit(bank, sig[split]), fgclip2_distribution=distribution_audit(bank, fg[split]))
    write_json(OUT / ds / "diagnostics.json", report)
    if ds == "cub": cub_audit(samples, bank, sig, fg)
    freeze(OUT / ds / "complete.json", [OUT / ds / "evidence_complete.json", OUT / ds / "diagnostics.json"] +
           ([OUT / ds / "reliability_audit.json"] if ds == "cub" else []))

def main():
    import torch, msvcrt
    p = argparse.ArgumentParser(); p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        preflight(); return
    with (OUT / "run.lock").open("a+b") as lock:
        lock.write(b"0"); lock.flush(); lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        encoder = FGClip2Encoder(batch_size=1)
        try:
            for ds in (args.dataset,) if args.dataset else DATASETS:
                event("prepare", dataset=ds)
                samples, bank, sig, fg = prepare(ds, encoder)
                diagnostics(ds, samples, bank, sig, fg)
                event("dataset_complete", dataset=ds)
                gc.collect()
            event("comparison_scoring_complete", next_stage="review evidence audit before choosing backend; Gemma first, Qwen2.5 second")
        except BaseException as error:
            event("failed", error=str(error)); raise

if __name__ == "__main__": main()

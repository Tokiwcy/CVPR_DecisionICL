from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .atoms import (
    collapse_exact_duplicate_groups,
    filter_bank,
    group_semantic_text,
    lint_groups,
    response_aware_deduplicate,
)
from .audit import (
    AuditThresholds,
    EvidenceAnnotation,
    assert_audit_passed,
    distribution_audit,
    make_blind_audit_bundle,
    score_evidence_annotations,
)
from .baselines import mmices_question_only, muier_classification_adapted
from .calibration import (
    CALIBRATION_OBJECTIVE,
    CalibrationArtifact,
    cross_fit_calibration,
    folds_hash,
    stratified_folds,
)
from .contracts import (
    MethodContract,
    assert_discovery_is_train_only,
    assert_test_retrieval_splits,
    manifest_hash,
)
from .discovery import (
    DEFAULT_PERSPECTIVES,
    TransformersDiscoveryBackend,
    make_discovery_requests,
    parse_discovery_response,
)
from .encoders import FlairEncoder, SiglipEncoder
from .evidence import (
    EmbeddingStore,
    EvidenceCalibration,
    EvidenceStore,
    apply_evidence_calibration,
    build_embedding_store,
    build_evidence_store,
    fit_train_evidence_calibration,
    make_response_fingerprints,
)
from .hashing import stable_hash
from .io import (
    append_jsonl,
    load_atom_bank,
    load_manifest,
    read_json,
    read_jsonl,
    save_atom_bank,
    write_json,
    write_jsonl,
)
from .metrics import accuracy_metrics, paired_comparison
from .plans import RetrievalPlan, make_plan
from .retrieval import GROUP_SIMILARITY, GLOBAL_SIMILARITY, random_retrieval, retrieve_full_pool
from .schema import AtomBank, Sample
from .vlm import TransformersChatVLM, make_classification_prompt, run_inference_plan


def _load_classes(path: str) -> tuple[str, ...]:
    value = read_json(path)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("classes JSON must be a list of strings")
    return tuple(value)


def _balanced_discovery_subset(samples: Sequence[Sample], budget: int, seed: int) -> list[Sample]:
    if budget < 2 or budget > len(samples):
        raise ValueError("discovery budget must be between 2 and the train size")
    by_label: dict[str, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)
    for label, rows in by_label.items():
        rows.sort(key=lambda row: stable_hash({"seed": seed, "label": label, "id": row.sample_id}))
    selected: list[Sample] = []
    rank = 0
    while len(selected) < budget:
        added = False
        for label in sorted(by_label):
            if rank < len(by_label[label]):
                selected.append(by_label[label][rank])
                added = True
                if len(selected) == budget:
                    break
        if not added:
            break
        rank += 1
    return sorted(selected, key=lambda sample: sample.sample_id)


def command_discover(args: argparse.Namespace) -> None:
    samples = load_manifest(args.train_manifest, expected_split="train")
    classes = _load_classes(args.classes)
    selected = _balanced_discovery_subset(samples, args.discovery_budget, args.seed)
    requests = make_discovery_requests(
        sample_ids=[sample.sample_id for sample in selected],
        image_paths=[sample.image_path for sample in selected],
        task_instruction=args.task_instruction,
        class_names=classes,
        batch_size=args.batch_size,
        max_groups_per_perspective=args.max_groups,
        perspectives=DEFAULT_PERSPECTIVES,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "discovery_responses.jsonl"
    cached_rows = read_jsonl(response_path) if response_path.exists() else []
    cache = {row["request_hash"]: row for row in cached_rows}
    backend = TransformersDiscoveryBackend(
        model_id=args.model,
        model_revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    groups = []
    rejected_responses: list[dict[str, Any]] = []
    for request in requests:
        request_hash = stable_hash(
            {
                "prompt_id": request.prompt_id,
                "model": args.model,
                "model_revision": args.model_revision,
                "device_independent_dtype": args.dtype,
                "max_new_tokens": args.max_new_tokens,
                "min_pixels": backend.min_pixels,
                "max_pixels": backend.max_pixels,
            }
        )
        if request_hash not in cache:
            raw = backend.generate(request.image_paths, request.prompt)
            cache[request_hash] = {
                "request_hash": request_hash,
                "prompt_id": request.prompt_id,
                "batch_id": request.batch_id,
                "sample_ids": list(request.sample_ids),
                "perspective": request.perspective,
                "raw_output": raw,
            }
            write_jsonl(response_path, [cache[key] for key in sorted(cache)])
        try:
            parsed_groups = parse_discovery_response(
                str(cache[request_hash]["raw_output"]),
                request=request,
                source_model=f"{args.model}@{args.model_revision}",
            )
        except (TypeError, ValueError) as error:
            rejected_responses.append(
                {
                    "request_hash": request_hash,
                    "prompt_id": request.prompt_id,
                    "batch_id": request.batch_id,
                    "perspective": request.perspective,
                    "error": str(error),
                }
            )
            continue
        groups.extend(parsed_groups)
    if not groups:
        raise RuntimeError("all discovery responses were rejected; no legal factor groups remain")
    bank = AtomBank(
        task_name=args.task_name,
        task_instruction=args.task_instruction,
        class_names=classes,
        factor_scope="task",
        groups=tuple(groups),
        discovery_split="train",
        discovery_seed=args.seed,
    )
    bank.validate()
    assert_discovery_is_train_only(bank, samples)
    save_atom_bank(output_dir / "atom_bank.raw.json", bank)
    write_json(output_dir / "atom_bank.raw.lint.json", [issue.to_dict() for issue in lint_groups(bank.groups)])
    exact_kept, exact_removed = collapse_exact_duplicate_groups(bank.groups)
    exact_bank = filter_bank(bank, exact_kept)
    save_atom_bank(output_dir / "atom_bank.exact.json", exact_bank)
    write_json(
        output_dir / "atom_bank.exact_dedup.json",
        {
            "raw_bank_hash": bank.bank_hash,
            "exact_bank_hash": exact_bank.bank_hash,
            "kept_group_ids": list(exact_kept),
            "removed_to_representative": exact_removed,
        },
    )
    write_json(output_dir / "discovery_rejections.json", rejected_responses)
    write_json(output_dir / "discovery_subset.json", [sample.to_dict() for sample in selected])


def command_lint(args: argparse.Namespace) -> None:
    bank = load_atom_bank(args.bank)
    issues = [issue.to_dict() for issue in lint_groups(bank.groups, max_states=args.max_states)]
    write_json(args.output, {"bank_hash": bank.bank_hash, "issues": issues})
    if any(issue["level"] == "error" for issue in issues):
        raise RuntimeError("atom bank lint contains errors")


def command_encode(args: argparse.Namespace) -> None:
    bank = load_atom_bank(args.bank)
    samples = load_manifest(args.manifest, expected_split=args.split)
    if args.encoder_backend == "siglip":
        encoder = SiglipEncoder(
            model_id=args.model,
            model_revision=args.model_revision,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
        )
    else:
        if not args.flair_source:
            raise ValueError("--flair-source is required for encoder-backend=flair")
        if not args.checkpoint_sha256:
            raise ValueError("--checkpoint-sha256 is required for encoder-backend=flair")
        encoder = FlairEncoder(
            model_id=args.model,
            source_dir=args.flair_source,
            model_revision=args.model_revision,
            checkpoint_sha256=args.checkpoint_sha256,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
        )
    embeddings = build_embedding_store(samples, encoder)
    raw_evidence = build_evidence_store(
        samples=samples,
        image_embeddings=embeddings,
        bank=bank,
        encoder=encoder,
        text_template=args.text_template,
    )
    if args.evidence_calibration:
        evidence_calibration = EvidenceCalibration.load(args.evidence_calibration)
    elif args.split == "train":
        evidence_calibration = fit_train_evidence_calibration(
            samples=samples,
            bank=bank,
            uncalibrated_store=raw_evidence,
        )
    else:
        raise ValueError("validation/test encoding requires the train --evidence-calibration artifact")
    evidence = apply_evidence_calibration(
        store=raw_evidence,
        bank=bank,
        calibration=evidence_calibration,
    )
    prefix = Path(args.output_prefix)
    embeddings.save(str(prefix) + ".embeddings")
    evidence.save(str(prefix) + ".evidence")
    if args.split == "train" and not args.evidence_calibration:
        evidence_calibration.save(str(prefix) + ".evidence_calibration.json")
    write_json(str(prefix) + ".distribution_audit.json", distribution_audit(bank, evidence))


def command_audit_bundle(args: argparse.Namespace) -> None:
    samples = load_manifest(args.manifest, expected_split="train")
    bank = load_atom_bank(args.bank)
    evidence = EvidenceStore.load(args.evidence_prefix)
    write_json(
        args.output,
        make_blind_audit_bundle(
            samples=samples,
            bank=bank,
            evidence=evidence,
            examples_per_band=args.examples_per_band,
            uniform_examples_per_group=args.uniform_examples_per_group,
            seed=args.seed,
        ),
    )


def command_audit_score(args: argparse.Namespace) -> None:
    bank = load_atom_bank(args.bank)
    evidence = EvidenceStore.load(args.evidence_prefix)
    annotations = [EvidenceAnnotation.from_dict(row) for row in read_jsonl(args.annotations)]
    thresholds = AuditThresholds(
        minimum_annotations_per_group=args.minimum_annotations,
        minimum_applicable_fraction=args.minimum_coverage,
        minimum_top1_state_accuracy=args.minimum_top1_accuracy,
        minimum_signed_direction_accuracy=args.minimum_direction_accuracy,
        minimum_ordinal_spearman=args.minimum_ordinal_spearman,
    )
    write_json(
        args.output,
        score_evidence_annotations(
            bank=bank,
            evidence=evidence,
            annotations=annotations,
            thresholds=thresholds,
        ),
    )


def command_deduplicate(args: argparse.Namespace) -> None:
    bank = load_atom_bank(args.bank)
    evidence = EvidenceStore.load(args.evidence_prefix)
    encoder = SiglipEncoder(
        model_id=args.model,
        model_revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
    semantic = encoder.encode_texts([group_semantic_text(group) for group in bank.groups])
    fingerprints = make_response_fingerprints(evidence, pair_count=args.pair_count, seed=args.seed)
    quality: dict[str, float] = {}
    if args.audit_report:
        audit = read_json(args.audit_report)
        if audit.get("bank_hash") != bank.bank_hash:
            raise ValueError("dedup quality audit belongs to a different bank")
        if audit.get("evidence_store_hash") != evidence.store_hash:
            raise ValueError("dedup quality audit belongs to a different evidence database")
        quality = {
            group_id: float(row.get("applicable_fraction", 0.0))
            * float(row.get("top1_state_accuracy", 0.0))
            for group_id, row in audit.get("groups", {}).items()
        }
    result = response_aware_deduplicate(
        group_ids=list(evidence.group_ids),
        semantic_embeddings=semantic,
        response_fingerprints=fingerprints,
        quality_scores=quality,
        semantic_threshold=args.semantic_threshold,
        response_threshold=args.response_threshold,
    )
    write_json(args.output_report, result.to_dict())
    save_atom_bank(args.output_bank, filter_bank(bank, result.kept_group_ids))


def _load_calibration_inputs(args: argparse.Namespace) -> tuple[Any, ...]:
    candidates = load_manifest(args.train_manifest, expected_split="train")
    validation = load_manifest(args.validation_manifest, expected_split="validation")
    bank = load_atom_bank(args.bank)
    candidate_embeddings = EmbeddingStore.load(args.train_embedding_prefix)
    validation_embeddings = EmbeddingStore.load(args.validation_embedding_prefix)
    candidate_evidence = EvidenceStore.load(args.train_evidence_prefix)
    validation_evidence = EvidenceStore.load(args.validation_evidence_prefix)
    return (
        candidates,
        validation,
        bank,
        candidate_embeddings,
        validation_embeddings,
        candidate_evidence,
        validation_evidence,
    )


def command_calibrate(args: argparse.Namespace) -> None:
    (
        candidates,
        validation,
        bank,
        candidate_embeddings,
        validation_embeddings,
        candidate_evidence,
        validation_evidence,
    ) = _load_calibration_inputs(args)
    if args.audit_report:
        assert_audit_passed(
            read_json(args.audit_report),
            bank.bank_hash,
            candidate_evidence.store_hash,
        )
    assignments = stratified_folds(validation, fold_count=args.folds, seed=args.seed)
    contract = MethodContract(
        atom_bank_hash=bank.bank_hash,
        evidence_model=candidate_evidence.model_id,
        evidence_model_revision=candidate_evidence.model_revision,
        evidence_text_template=candidate_evidence.text_template,
        evidence_transform=candidate_evidence.transform,
        group_similarity=GROUP_SIMILARITY,
        global_similarity=GLOBAL_SIMILARITY,
        calibration_objective=CALIBRATION_OBJECTIVE,
        calibration_folds_hash=folds_hash(validation, assignments),
        candidate_manifest_hash=manifest_hash(candidates),
        calibration_manifest_hash=manifest_hash(validation),
        candidate_embedding_hash=candidate_embeddings.store_hash,
        calibration_embedding_hash=validation_embeddings.store_hash,
        candidate_evidence_hash=candidate_evidence.store_hash,
        calibration_evidence_hash=validation_evidence.store_hash,
    )
    artifact = cross_fit_calibration(
        candidate_samples=candidates,
        validation_samples=validation,
        candidate_embeddings=candidate_embeddings,
        validation_embeddings=validation_embeddings,
        candidate_evidence=candidate_evidence,
        validation_evidence=validation_evidence,
        contract=contract,
        fold_count=args.folds,
        seed=args.seed,
    )
    write_json(args.output, artifact.to_dict())


def _verify_train_contract(
    artifact: CalibrationArtifact,
    train_samples: Sequence[Sample],
    train_embeddings: EmbeddingStore,
    train_evidence: EvidenceStore,
) -> None:
    contract = artifact.contract
    checks = {
        "candidate_manifest_hash": manifest_hash(train_samples),
        "candidate_embedding_hash": train_embeddings.store_hash,
        "candidate_evidence_hash": train_evidence.store_hash,
    }
    for name, actual in checks.items():
        if getattr(contract, name) != actual:
            raise ValueError(f"calibration artifact contract mismatch: {name}")
    if artifact.group_ids != train_evidence.group_ids:
        raise ValueError("calibration artifact group order mismatch")


def command_retrieve(args: argparse.Namespace) -> None:
    train_samples = load_manifest(args.train_manifest, expected_split="train")
    query_samples = load_manifest(args.query_manifest, expected_split=args.query_split)
    if args.query_split == "test":
        assert_test_retrieval_splits(train_samples, query_samples)
    train_embeddings = EmbeddingStore.load(args.train_embedding_prefix)
    query_embeddings = EmbeddingStore.load(args.query_embedding_prefix)
    train_evidence = EvidenceStore.load(args.train_evidence_prefix)
    query_evidence = EvidenceStore.load(args.query_evidence_prefix)
    if args.method == "ours":
        if not args.calibration:
            raise ValueError("--calibration is required for method=ours")
        artifact = CalibrationArtifact.from_dict(read_json(args.calibration))
        _verify_train_contract(artifact, train_samples, train_embeddings, train_evidence)
        if query_evidence.bank_hash != artifact.contract.atom_bank_hash:
            raise ValueError("query evidence uses a different atom bank")
        if (
            query_evidence.model_id != artifact.contract.evidence_model
            or query_evidence.model_revision != artifact.contract.evidence_model_revision
            or query_evidence.text_template != artifact.contract.evidence_text_template
            or query_evidence.transform != artifact.contract.evidence_transform
        ):
            raise ValueError("query evidence method contract mismatch")
        result = retrieve_full_pool(
            weights=np.asarray(artifact.deployment_weights),
            query_embeddings=query_embeddings,
            candidate_embeddings=train_embeddings,
            query_evidence=query_evidence,
            candidate_evidence=train_evidence,
            shots=args.shots,
        )
        contract_hash = artifact.contract.contract_hash
    elif args.method == "rices":
        weights = np.zeros(1 + len(train_evidence.group_ids), dtype=np.float64)
        weights[0] = 1.0
        result = retrieve_full_pool(
            weights=weights,
            query_embeddings=query_embeddings,
            candidate_embeddings=train_embeddings,
            query_evidence=query_evidence,
            candidate_evidence=train_evidence,
            shots=args.shots,
        )
        contract_hash = stable_hash(
            {"method": "rices", "embedding_hash": train_embeddings.store_hash, "shots": args.shots}
        )
    elif args.method == "mmices-question":
        if not args.model:
            raise ValueError("MMICES requires --model for question-text embeddings")
        encoder = SiglipEncoder(
            model_id=args.model,
            model_revision=args.model_revision,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
        )
        result = mmices_question_only(
            query_embeddings=query_embeddings,
            candidate_embeddings=train_embeddings,
            query_questions=[sample.question for sample in query_samples],
            candidate_questions=[sample.question for sample in train_samples],
            text_encoder=encoder,
            candidate_k=args.candidate_k,
            shots=args.shots,
        )
        contract_hash = stable_hash(
            {
                "method": "mmices-question-only",
                "candidate_k": args.candidate_k,
                "shots": args.shots,
                "tie_policy": "stable-rices-order",
                "model": args.model,
                "model_revision": args.model_revision,
            }
        )
    elif args.method == "muier":
        if not args.model:
            raise ValueError("MUIER requires --model for train-label text embeddings")
        encoder = SiglipEncoder(
            model_id=args.model,
            model_revision=args.model_revision,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
        )
        result = muier_classification_adapted(
            query_embeddings=query_embeddings,
            candidate_embeddings=train_embeddings,
            candidate_label_texts=[f"an image of {sample.label}" for sample in train_samples],
            text_encoder=encoder,
            shots=args.shots,
        )
        contract_hash = stable_hash(
            {
                "method": "muier-classification-adapted",
                "shots": args.shots,
                "model": args.model,
                "model_revision": args.model_revision,
                "memory_text": "known-train-label",
            }
        )
    elif args.method == "random":
        result = random_retrieval(
            query_ids=[sample.sample_id for sample in query_samples],
            candidate_ids=[sample.sample_id for sample in train_samples],
            shots=args.shots,
            seed=args.seed,
        )
        contract_hash = stable_hash({"method": "random", "seed": args.seed, "shots": args.shots})
    else:
        raise ValueError(f"unsupported retrieval method: {args.method}")
    plan = make_plan(args.method, contract_hash, result)
    plan.validate(query_samples, train_samples)
    write_json(args.output, plan.to_dict())


def command_infer(args: argparse.Namespace) -> None:
    train_samples = load_manifest(args.train_manifest, expected_split="train")
    query_samples = load_manifest(args.query_manifest, expected_split=args.query_split)
    plan = RetrievalPlan.from_dict(read_json(args.plan))
    classes = _load_classes(args.classes)
    prompt = make_classification_prompt(args.task_instruction, classes)
    backend = TransformersChatVLM(
        model_id=args.model,
        model_revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    cache_path = Path(args.cache)
    cache_rows = read_jsonl(cache_path) if cache_path.exists() else []
    cache: dict[str, dict[str, object]] = {}
    for row in cache_rows:
        request_hash = str(row["request_hash"])
        if request_hash in cache:
            raise ValueError(f"duplicate request_hash in inference cache: {request_hash}")
        cache[request_hash] = dict(row["value"])
    persisted = set(cache)

    def checkpoint(current: dict[str, dict[str, object]]) -> None:
        new_keys = [key for key in current if key not in persisted]
        append_jsonl(cache_path, ({"request_hash": key, "value": current[key]} for key in new_keys))
        persisted.update(new_keys)
    rows, updated_cache = run_inference_plan(
        plan=plan,
        queries=query_samples,
        candidates=train_samples,
        prompt=prompt,
        backend=backend,
        cache=cache,
        checkpoint=checkpoint,
    )
    if set(updated_cache) != persisted:
        raise RuntimeError("inference cache checkpoint did not persist every request")
    write_jsonl(args.output, rows)


def _prediction_map(path: str) -> dict[str, str]:
    rows = read_jsonl(path)
    result: dict[str, str] = {}
    for row in rows:
        query_id = str(row["query_id"])
        if query_id in result:
            raise ValueError(f"duplicate prediction query_id in {path}: {query_id}")
        result[query_id] = str(row["prediction_label"])
    return result


def command_evaluate(args: argparse.Namespace) -> None:
    samples = load_manifest(args.test_manifest, expected_split="test")
    methods: dict[str, dict[str, str]] = {}
    for item in args.predictions:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise ValueError("--predictions entries must be NAME=PATH")
        methods[name] = _prediction_map(path)
    query_ids = [sample.sample_id for sample in samples]
    labels = [sample.label for sample in samples]
    for name, mapping in methods.items():
        if set(mapping) != set(query_ids):
            raise ValueError(f"prediction coverage mismatch for {name}")
    report: dict[str, Any] = {
        "n": len(samples),
        "metrics": {
            name: accuracy_metrics(labels, [mapping[query_id] for query_id in query_ids])
            for name, mapping in methods.items()
        },
        "paired_vs_baseline": {},
    }
    if args.baseline not in methods:
        raise ValueError("--baseline must name one supplied prediction file")
    baseline = [methods[args.baseline][query_id] for query_id in query_ids]
    for name, mapping in methods.items():
        if name == args.baseline:
            continue
        report["paired_vs_baseline"][name] = paired_comparison(
            labels=labels,
            baseline_predictions=baseline,
            method_predictions=[mapping[query_id] for query_id in query_ids],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    write_json(args.output, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dgdr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover")
    discover.add_argument("--train-manifest", required=True)
    discover.add_argument("--classes", required=True)
    discover.add_argument("--task-name", required=True)
    discover.add_argument("--task-instruction", required=True)
    discover.add_argument("--model", required=True)
    discover.add_argument("--model-revision", default="local")
    discover.add_argument("--device", default="cuda")
    discover.add_argument("--dtype", default="bfloat16")
    discover.add_argument("--discovery-budget", type=int, default=60)
    discover.add_argument("--batch-size", type=int, default=6)
    discover.add_argument("--max-groups", type=int, default=6)
    discover.add_argument("--max-new-tokens", type=int, default=1536)
    discover.add_argument("--min-pixels", type=int, default=65536)
    discover.add_argument("--max-pixels", type=int, default=401408)
    discover.add_argument("--seed", type=int, default=42)
    discover.add_argument("--output-dir", required=True)
    discover.set_defaults(handler=command_discover)

    lint = subparsers.add_parser("lint-bank")
    lint.add_argument("--bank", required=True)
    lint.add_argument("--max-states", type=int, default=6)
    lint.add_argument("--output", required=True)
    lint.set_defaults(handler=command_lint)

    encode = subparsers.add_parser("encode-split")
    encode.add_argument("--bank", required=True)
    encode.add_argument("--manifest", required=True)
    encode.add_argument("--split", choices=("train", "validation", "test"), required=True)
    encode.add_argument("--model", required=True)
    encode.add_argument("--encoder-backend", choices=("siglip", "flair"), default="siglip")
    encode.add_argument("--flair-source")
    encode.add_argument("--checkpoint-sha256")
    encode.add_argument("--model-revision", default="local")
    encode.add_argument("--device", default="cuda")
    encode.add_argument("--dtype", default="float16")
    encode.add_argument("--batch-size", type=int, default=32)
    encode.add_argument("--text-template", default="This image shows {description}.")
    encode.add_argument("--evidence-calibration")
    encode.add_argument("--output-prefix", required=True)
    encode.set_defaults(handler=command_encode)

    bundle = subparsers.add_parser("make-audit-bundle")
    bundle.add_argument("--bank", required=True)
    bundle.add_argument("--manifest", required=True)
    bundle.add_argument("--evidence-prefix", required=True)
    bundle.add_argument("--examples-per-band", type=int, default=1)
    bundle.add_argument("--uniform-examples-per-group", type=int, default=3)
    bundle.add_argument("--seed", type=int, default=42)
    bundle.add_argument("--output", required=True)
    bundle.set_defaults(handler=command_audit_bundle)

    score = subparsers.add_parser("score-audit")
    score.add_argument("--bank", required=True)
    score.add_argument("--evidence-prefix", required=True)
    score.add_argument("--annotations", required=True)
    score.add_argument("--minimum-annotations", type=int, default=6)
    score.add_argument("--minimum-coverage", type=float, default=0.80)
    score.add_argument("--minimum-top1-accuracy", type=float, default=0.70)
    score.add_argument("--minimum-direction-accuracy", type=float, default=0.70)
    score.add_argument("--minimum-ordinal-spearman", type=float, default=0.50)
    score.add_argument("--output", required=True)
    score.set_defaults(handler=command_audit_score)

    dedup = subparsers.add_parser("deduplicate-bank")
    dedup.add_argument("--bank", required=True)
    dedup.add_argument("--evidence-prefix", required=True)
    dedup.add_argument("--model", required=True)
    dedup.add_argument("--model-revision", default="local")
    dedup.add_argument("--device", default="cuda")
    dedup.add_argument("--dtype", default="float16")
    dedup.add_argument("--batch-size", type=int, default=32)
    dedup.add_argument("--pair-count", type=int, default=4096)
    dedup.add_argument("--seed", type=int, default=42)
    dedup.add_argument("--semantic-threshold", type=float, default=0.90)
    dedup.add_argument("--response-threshold", type=float, default=0.98)
    dedup.add_argument("--audit-report")
    dedup.add_argument("--output-report", required=True)
    dedup.add_argument("--output-bank", required=True)
    dedup.set_defaults(handler=command_deduplicate)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--bank", required=True)
    calibrate.add_argument("--train-manifest", required=True)
    calibrate.add_argument("--validation-manifest", required=True)
    calibrate.add_argument("--train-embedding-prefix", required=True)
    calibrate.add_argument("--validation-embedding-prefix", required=True)
    calibrate.add_argument("--train-evidence-prefix", required=True)
    calibrate.add_argument("--validation-evidence-prefix", required=True)
    calibrate.add_argument("--audit-report")
    calibrate.add_argument("--folds", type=int, default=5)
    calibrate.add_argument("--seed", type=int, default=42)
    calibrate.add_argument("--output", required=True)
    calibrate.set_defaults(handler=command_calibrate)

    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument(
        "--method",
        choices=("ours", "rices", "random", "mmices-question", "muier"),
        required=True,
    )
    retrieve.add_argument("--calibration")
    retrieve.add_argument("--train-manifest", required=True)
    retrieve.add_argument("--query-manifest", required=True)
    retrieve.add_argument("--query-split", choices=("validation", "test"), required=True)
    retrieve.add_argument("--train-embedding-prefix", required=True)
    retrieve.add_argument("--query-embedding-prefix", required=True)
    retrieve.add_argument("--train-evidence-prefix", required=True)
    retrieve.add_argument("--query-evidence-prefix", required=True)
    retrieve.add_argument("--shots", type=int, default=1)
    retrieve.add_argument("--candidate-k", type=int, default=32)
    retrieve.add_argument("--model")
    retrieve.add_argument("--model-revision", default="local")
    retrieve.add_argument("--device", default="cuda")
    retrieve.add_argument("--dtype", default="float16")
    retrieve.add_argument("--batch-size", type=int, default=32)
    retrieve.add_argument("--seed", type=int, default=42)
    retrieve.add_argument("--output", required=True)
    retrieve.set_defaults(handler=command_retrieve)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--plan", required=True)
    infer.add_argument("--train-manifest", required=True)
    infer.add_argument("--query-manifest", required=True)
    infer.add_argument("--query-split", choices=("validation", "test"), required=True)
    infer.add_argument("--classes", required=True)
    infer.add_argument("--task-instruction", required=True)
    infer.add_argument("--model", required=True)
    infer.add_argument("--model-revision", default="local")
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--dtype", default="bfloat16")
    infer.add_argument("--max-new-tokens", type=int, default=8)
    infer.add_argument("--cache", required=True)
    infer.add_argument("--output", required=True)
    infer.set_defaults(handler=command_infer)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--test-manifest", required=True)
    evaluate.add_argument("--predictions", action="append", required=True)
    evaluate.add_argument("--baseline", default="rices")
    evaluate.add_argument("--bootstrap-samples", type=int, default=10_000)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

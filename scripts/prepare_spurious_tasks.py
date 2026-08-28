from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
QWEN4B = "Qwen/Qwen3-VL-4B-Instruct"
SIGLIP2 = "google/siglip2-base-patch16-224"
SEED = 20260825
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _ranked_files(
    folder: Path, salt: str, *, excluded: set[str] | None = None
) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing image folder: {folder}")
    excluded = excluded or set()
    files = [
        path
        for path in folder.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
        and str(path.resolve()).casefold() not in excluded
    ]
    return sorted(
        files,
        key=lambda path: hashlib.sha256(
            f"{SEED}:{salt}:{path.name}".encode("utf-8")
        ).digest(),
    )


def _take(files: Sequence[Path], count: int, *, offset: int = 0, name: str) -> list[Path]:
    chosen = list(files[offset : offset + count])
    if len(chosen) != count:
        raise ValueError(
            f"{name} needs {count} images at offset {offset}, found {len(chosen)}"
        )
    return chosen


def _row(
    *,
    task: str,
    split: str,
    image: Path,
    label: str,
    spurious_attribute: str,
    aligned_attribute: str,
    source_split: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    conflict = spurious_attribute != aligned_attribute
    safe_stem = image.stem.replace(" ", "_")
    path_token = hashlib.sha256(str(image.resolve()).encode("utf-8")).hexdigest()[:12]
    sample_id = (
        f"{task}-{split}-{label.replace(' ', '_')}-{spurious_attribute}-"
        f"{path_token}-{safe_stem}"
    )
    metadata: dict[str, Any] = {
        "diagnostic": conflict,
        "conflict": conflict,
        "spurious_attribute": spurious_attribute,
        "aligned_attribute": aligned_attribute,
        "source_split": source_split,
    }
    if extra:
        metadata.update(extra)
    return {
        "id": sample_id,
        "split": split,
        "image": str(image.resolve()),
        "text": "",
        "label": label,
        "group": f"{label}_on_{spurious_attribute}",
        "metadata": metadata,
    }


def _write_task(
    output: Path,
    rows: list[dict[str, Any]],
    labels: Sequence[str],
    *,
    task_instruction: str,
    diagnostic_definition: str,
) -> None:
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, value in Counter(ids).items() if value > 1]
        raise ValueError(f"Duplicate manifest IDs: {duplicates[:5]}")
    rows.sort(key=lambda row: (row["split"], row["id"]))
    manifest = output / "manifest.jsonl"
    _write_jsonl(manifest, rows)
    config = {
        "seed": SEED,
        "manifest": str(manifest.resolve()),
        "output_dir": str((output / "unused_source_run").resolve()),
        "labels": list(labels),
        "selection_method_name": "weighted_counter_spurious_global_3shot",
        "diagnostic": {
            "kind": "manifest_boolean",
            "metadata_key": "diagnostic",
            "definition": diagnostic_definition,
        },
        "lambda_selection_objective": "group_robustness",
        # Mean-group accuracy already gives aligned and conflict groups equal weight.
        # Keep the guard effectively disabled so lambda is learned from that balanced
        # validation objective instead of being forced toward RICES.
        "max_aligned_accuracy_drop": 1.0,
        "target_model": {
            "kind": "qwen3_vl",
            "model": QWEN4B,
            "dtype": "bfloat16",
            "device_map": "auto",
            "attn_implementation": "sdpa",
            "local_files_only": False,
            "classification_batch_size": 4,
            "icl_batch_size": 2,
            "min_pixels": 65536,
            "max_pixels": 401408,
            "efficient_next_logits": True,
            "max_new_tokens": 32,
            "task_instruction": task_instruction,
        },
        "similarity_model": {
            "kind": "siglip2",
            "model": SIGLIP2,
            "dtype": "float16",
            "device": "cuda",
            "batch_size": 48,
            "local_files_only": False,
        },
    }
    config_path = output / "qwen3vl4b_bf16.yaml"
    output.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (output / "labels.json").write_text(
        json.dumps(list(labels), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    by_split = Counter(str(row["split"]) for row in rows)
    by_group = defaultdict(Counter)
    for row in rows:
        by_group[str(row["split"])][str(row["group"])] += 1
    summary = {
        "manifest": str(manifest),
        "config": str(config_path),
        "counts": dict(sorted(by_split.items())),
        "groups": {split: dict(sorted(groups.items())) for split, groups in by_group.items()},
        "diagnostic_train_rate": sum(
            bool(row["metadata"]["diagnostic"])
            for row in rows
            if row["split"] == "train"
        )
        / by_split["train"],
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def prepare_spuco_animals(args: argparse.Namespace) -> None:
    labels = ["landbird", "waterbird", "small dog", "big dog"]
    groups = [
        ("spuco_birds", "landbirds", "land", "landbird", "land"),
        ("spuco_birds", "landbirds", "water", "landbird", "land"),
        ("spuco_birds", "waterbirds", "land", "waterbird", "water"),
        ("spuco_birds", "waterbirds", "water", "waterbird", "water"),
        ("spuco_dogs", "small_dogs", "indoor", "small dog", "indoor"),
        ("spuco_dogs", "small_dogs", "outdoor", "small dog", "indoor"),
        ("spuco_dogs", "big_dogs", "indoor", "big dog", "outdoor"),
        ("spuco_dogs", "big_dogs", "outdoor", "big dog", "outdoor"),
    ]
    rows: list[dict[str, Any]] = []
    for dataset, raw_label, attribute, label, aligned in groups:
        diagnostic = attribute != aligned
        train_count = (
            args.train_minority_per_group if diagnostic else args.train_majority_per_group
        )
        for source_split, target_split, count in (
            ("train", "train", train_count),
            ("val", "validation", args.validation_per_group),
            ("test", "test", args.test_per_group),
        ):
            folder = args.source / dataset / source_split / raw_label / attribute
            files = _ranked_files(folder, f"{dataset}:{source_split}:{raw_label}:{attribute}")
            selected = (
                files
                if args.preserve_source_splits
                else _take(
                    files, count, name=f"{source_split}/{raw_label}/{attribute}"
                )
            )
            for image in selected:
                rows.append(
                    _row(
                        task="spuco",
                        split=target_split,
                        image=image,
                        label=label,
                        spurious_attribute=attribute,
                        aligned_attribute=aligned,
                        source_split=source_split,
                    )
                )
    _write_task(
        args.output,
        rows,
        labels,
        task_instruction="Identify the animal category shown in the image.",
        diagnostic_definition=(
            "known counter-spurious training group: landbird on water, waterbird on land, "
            "small dog outdoors, or big dog indoors"
        ),
    )


def _load_metashift_groups(path: Path) -> dict[str, str]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    result: dict[str, str] = {}
    for image_id, value in raw.items():
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"Ambiguous MetaShift group for {image_id}: {value}")
            value = value[0]
        group = str(value)
        if group not in {"cat(indoor)", "cat(outdoor)", "dog(indoor)", "dog(outdoor)"}:
            raise ValueError(f"Unexpected MetaShift group: {group}")
        result[str(image_id)] = group
    return result


def prepare_metashift(args: argparse.Namespace) -> None:
    group_map = _load_metashift_groups(args.source / "imageID_to_group.pkl")
    labels = ["cat", "dog"]
    expected = {"cat": "indoor", "dog": "outdoor"}
    candidates: dict[tuple[str, str], list[Path]] = defaultdict(list)
    evaluation: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for source_split, destination in (
        ("train", candidates),
        ("val_out_of_domain", evaluation),
    ):
        for label in labels:
            folder = args.source / source_split / label
            for image in _ranked_files(folder, f"metashift:{source_split}:{label}"):
                group = group_map.get(image.stem)
                if group is None:
                    raise ValueError(f"MetaShift group metadata missing for {image.name}")
                group_label, raw_attribute = group[:-1].split("(", 1)
                if group_label != label:
                    raise ValueError(f"MetaShift label mismatch for {image.name}: {group}")
                destination[(label, raw_attribute)].append(image)
    rows: list[dict[str, Any]] = []
    for label in labels:
        for attribute in ("indoor", "outdoor"):
            diagnostic = attribute != expected[label]
            train_count = (
                args.train_minority_per_group
                if diagnostic
                else args.train_majority_per_group
            )
            train_files = sorted(
                candidates[(label, attribute)],
                key=lambda path: hashlib.sha256(path.name.encode()).digest(),
            )
            eval_files = sorted(
                evaluation[(label, attribute)],
                key=lambda path: hashlib.sha256(path.name.encode()).digest(),
            )
            for image in _take(train_files, train_count, name=f"train/{label}/{attribute}"):
                rows.append(
                    _row(
                        task="metashift",
                        split="train",
                        image=image,
                        label=label,
                        spurious_attribute=attribute,
                        aligned_attribute=expected[label],
                        source_split="train",
                    )
                )
            needed = args.validation_per_group + args.test_per_group
            selected = _take(eval_files, needed, name=f"eval/{label}/{attribute}")
            for split, subset in (
                ("validation", selected[: args.validation_per_group]),
                ("test", selected[args.validation_per_group :]),
            ):
                for image in subset:
                    rows.append(
                        _row(
                            task="metashift",
                            split=split,
                            image=image,
                            label=label,
                            spurious_attribute=attribute,
                            aligned_attribute=expected[label],
                            source_split="val_out_of_domain",
                        )
                    )
    _write_task(
        args.output,
        rows,
        labels,
        task_instruction="Identify whether the animal is a cat or a dog.",
        diagnostic_definition=(
            "known counter-spurious training group: cat outdoors or dog indoors"
        ),
    )


SPAWRIOUS_PROTOCOLS = {
    "o2o_easy": {
        "aligned": ["desert", "jungle", "dirt", "snow"],
        "test": ["dirt", "snow", "desert", "jungle"],
        "filler": "beach",
    },
    "o2o_medium": {
        "aligned": ["mountain", "beach", "dirt", "jungle"],
        "test": ["jungle", "dirt", "beach", "snow"],
        "filler": "desert",
    },
    "o2o_hard": {
        "aligned": ["jungle", "mountain", "snow", "desert"],
        "test": ["mountain", "snow", "desert", "jungle"],
        "filler": "beach",
    },
}


def prepare_spawrious(args: argparse.Namespace) -> None:
    labels = ["bulldog", "dachshund", "labrador", "corgi"]
    protocol = SPAWRIOUS_PROTOCOLS[args.challenge]
    aligned_by_label = dict(zip(labels, protocol["aligned"], strict=True))
    test_by_label = dict(zip(labels, protocol["test"], strict=True))
    root = args.source / "spawrious224"
    full_evaluation = bool(getattr(args, "full_evaluation", False))
    excluded_images = {
        str(Path(value).resolve()).casefold()
        for value in getattr(args, "exclude_image", [])
    }
    legacy_aligned = getattr(args, "train_aligned_per_environment", None)
    legacy_diagnostic = getattr(args, "train_diagnostic_per_environment", None)
    aligned_counts = (
        [legacy_aligned, legacy_aligned]
        if legacy_aligned is not None
        else [args.train_aligned_environment_0, args.train_aligned_environment_1]
    )
    diagnostic_counts = (
        [legacy_diagnostic, legacy_diagnostic]
        if legacy_diagnostic is not None
        else [
            args.train_diagnostic_environment_0,
            args.train_diagnostic_environment_1,
        ]
    )
    rows: list[dict[str, Any]] = []
    for label in labels:
        aligned = aligned_by_label[label]
        filler = str(protocol["filler"])
        test_attribute = test_by_label[label]
        for environment in (0, 1):
            aligned_files = _ranked_files(
                root / str(environment) / aligned / label,
                f"spawrious:{args.challenge}:{environment}:{aligned}:{label}",
                excluded=excluded_images,
            )
            diagnostic_files = _ranked_files(
                root / str(environment) / filler / label,
                f"spawrious:{args.challenge}:{environment}:{filler}:{label}",
                excluded=excluded_images,
            )
            if full_evaluation:
                aligned_fraction = 0.97 if environment == 0 else 0.87
                aligned_count = int(aligned_fraction * len(aligned_files))
                diagnostic_count = len(diagnostic_files) - int(
                    aligned_fraction * len(diagnostic_files)
                )
            else:
                aligned_count = int(aligned_counts[environment])
                diagnostic_count = int(diagnostic_counts[environment])
            for image in _take(
                aligned_files,
                aligned_count,
                name=f"aligned train env{environment}/{label}",
            ):
                rows.append(
                    _row(
                        task="spawrious",
                        split="train",
                        image=image,
                        label=label,
                        spurious_attribute=aligned,
                        aligned_attribute=aligned,
                        source_split=f"train_environment_{environment}",
                        extra={
                            "challenge": args.challenge,
                            "environment": environment,
                            "evaluation_mode": (
                                "full_official_ood" if full_evaluation else "pilot_strict_env"
                            ),
                        },
                    )
                )
            for image in _take(
                diagnostic_files,
                diagnostic_count,
                name=f"diagnostic train env{environment}/{label}",
            ):
                rows.append(
                    _row(
                        task="spawrious",
                        split="train",
                        image=image,
                        label=label,
                        spurious_attribute=filler,
                        aligned_attribute=aligned,
                        source_split=f"train_environment_{environment}",
                        extra={
                            "challenge": args.challenge,
                            "environment": environment,
                            "evaluation_mode": (
                                "full_official_ood" if full_evaluation else "pilot_strict_env"
                            ),
                        },
                    )
                )
            aligned_offset = aligned_count
            target_split = "validation" if environment == 0 else "test"
            aligned_query_count = (
                args.validation_per_group_per_environment
                if full_evaluation
                else args.query_per_group
            )
            for image in _take(
                aligned_files,
                aligned_query_count,
                offset=aligned_offset,
                name=f"aligned validation env{environment}/{label}",
            ):
                rows.append(
                    _row(
                        task="spawrious",
                        split="validation" if full_evaluation else target_split,
                        image=image,
                        label=label,
                        spurious_attribute=aligned,
                        aligned_attribute=aligned,
                        source_split=f"heldout_environment_{environment}",
                        extra={
                            "challenge": args.challenge,
                            "environment": environment,
                            "evaluation_mode": (
                                "full_official_ood" if full_evaluation else "pilot_strict_env"
                            ),
                        },
                    )
                )
            conflict_files = _ranked_files(
                root / str(environment) / test_attribute / label,
                f"spawrious:{args.challenge}:{environment}:{test_attribute}:{label}:ood",
                excluded=excluded_images,
            )
            conflict_offset = 0
            if test_attribute == filler:
                conflict_offset = diagnostic_count
            if test_attribute == aligned:
                conflict_offset = aligned_count + aligned_query_count
            if full_evaluation:
                validation_count = int(args.validation_per_group_per_environment)
                conflict_splits = (
                    (
                        "validation",
                        _take(
                            conflict_files,
                            validation_count,
                            offset=conflict_offset,
                            name=f"conflict validation env{environment}/{label}",
                        ),
                    ),
                    ("test", list(conflict_files[conflict_offset + validation_count :])),
                )
            else:
                conflict_splits = (
                    (
                        target_split,
                        _take(
                            conflict_files,
                            args.query_per_group,
                            offset=conflict_offset,
                            name=f"conflict {target_split}/{label}",
                        ),
                    ),
                )
            for split, split_files in conflict_splits:
                for image in split_files:
                    rows.append(
                        _row(
                            task="spawrious",
                            split=split,
                            image=image,
                            label=label,
                            spurious_attribute=test_attribute,
                            aligned_attribute=aligned,
                            source_split=f"ood_environment_{environment}",
                            extra={
                                "challenge": args.challenge,
                                "environment": environment,
                                "evaluation_mode": (
                                    "full_official_ood"
                                    if full_evaluation
                                    else "pilot_strict_env"
                                ),
                            },
                        )
                    )
    _write_task(
        args.output,
        rows,
        labels,
        task_instruction="Identify the dog breed shown in the image.",
        diagnostic_definition=(
            f"known counter-spurious training group for Spawrious {args.challenge}: "
            "the dog breed appears in the shared filler location instead of its aligned location"
        ),
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare spurious-correlation ICL tasks")
    sub = parser.add_subparsers(dest="task", required=True)

    spuco = sub.add_parser("spuco_animals")
    _add_common(spuco)
    spuco.add_argument("--train-majority-per-group", type=int, default=400)
    spuco.add_argument("--train-minority-per-group", type=int, default=20)
    spuco.add_argument("--validation-per-group", type=int, default=5)
    spuco.add_argument("--test-per-group", type=int, default=10)
    spuco.add_argument(
        "--preserve-source-splits",
        action="store_true",
        help=(
            "include every official train/val/test image without group-balanced "
            "subsampling; this preserves the source validation distribution"
        ),
    )

    metashift = sub.add_parser("metashift")
    _add_common(metashift)
    metashift.add_argument("--train-majority-per-group", type=int, default=400)
    metashift.add_argument("--train-minority-per-group", type=int, default=50)
    metashift.add_argument("--validation-per-group", type=int, default=20)
    metashift.add_argument("--test-per-group", type=int, default=20)

    spawrious = sub.add_parser("spawrious")
    _add_common(spawrious)
    spawrious.add_argument(
        "--challenge", choices=tuple(SPAWRIOUS_PROTOCOLS), default="o2o_easy"
    )
    spawrious.add_argument(
        "--train-aligned-per-environment",
        type=int,
        default=None,
        help="legacy override that uses the same aligned count in both environments",
    )
    spawrious.add_argument(
        "--train-diagnostic-per-environment",
        type=int,
        default=None,
        help="legacy override that uses the same diagnostic count in both environments",
    )
    spawrious.add_argument("--train-aligned-environment-0", type=int, default=97)
    spawrious.add_argument("--train-diagnostic-environment-0", type=int, default=3)
    spawrious.add_argument("--train-aligned-environment-1", type=int, default=87)
    spawrious.add_argument("--train-diagnostic-environment-1", type=int, default=13)
    spawrious.add_argument("--query-per-group", type=int, default=10)
    spawrious.add_argument(
        "--full-evaluation",
        action="store_true",
        help=(
            "use the complete official 97/3 and 87/13 training pools, hold out a "
            "balanced validation subset in both environments, and put every remaining "
            "OOD image in test"
        ),
    )
    spawrious.add_argument(
        "--validation-per-group-per-environment", type=int, default=50
    )
    spawrious.add_argument(
        "--exclude-image",
        action="append",
        default=[],
        help="absolute or relative image path to exclude after integrity verification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    if args.task == "spuco_animals":
        prepare_spuco_animals(args)
    elif args.task == "metashift":
        prepare_metashift(args)
    else:
        prepare_spawrious(args)


if __name__ == "__main__":
    main()

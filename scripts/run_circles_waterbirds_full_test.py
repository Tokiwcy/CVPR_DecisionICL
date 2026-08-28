from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from decision_atoms.backends import build_retriever, build_target
from decision_atoms.config import load_config
from decision_atoms.io import load_jsonl, load_manifest, write_json, write_jsonl
from decision_atoms.math_utils import normalize_rows
from decision_atoms.types import Sample
from run_self_discovered_diagnostic_retrieval import (
    append_jsonl,
    file_sha256,
    load_safe_train_and_test,
    predict_selection_rows,
    top_indices,
)
from run_self_discovered_multishot_full_test import paired_delta_statistics, parse_shots


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "waterbirds_self_discovered_qwen3vl4b_bf16.yaml"
DEFAULT_SOURCE = ROOT / "artifacts" / "waterbirds_self_discovered_full"
DEFAULT_BASE = ROOT / "artifacts" / "waterbirds_self_discovered_multishot_full_test"
DEFAULT_OUTPUT = ROOT / "artifacts" / "waterbirds_circles_multishot_full_test"
CIRCLES_PROMPT_ORDER = "descending_similarity_highest_first"
REFERENCE_PROMPT_ORDER = "ascending_similarity_most_similar_nearest_query"


def circles_budget(total_shots: int) -> tuple[int, int]:
    """Split an equal total image budget between IR and composed retrieval.

    The official CIRCLES default uses 16 original plus 16 composed examples.
    For odd small-budget diagnostics, the extra slot stays with standard image
    retrieval so a 1-shot run is exactly the RICES sanity check.
    """

    if total_shots <= 0:
        raise ValueError("total_shots must be positive")
    return (total_shots + 1) // 2, total_shots // 2


def parse_attribute(text: str) -> str:
    tail = text.split("### Attributes")[-1]
    for raw in tail.splitlines():
        value = raw.strip().lstrip("-*• ").strip()
        while value and value[0].isdigit():
            value = value[1:].lstrip(".):- ")
        if value:
            return value[:160]
    return ""


def parse_caption(text: str) -> str:
    value = text.strip()
    if "Target Image Description" in value:
        value = value.split("Target Image Description")[-1]
    return value.strip(" \n\r\t\"':{}")[:600]


def _batch_generate(
    target: Any,
    contents: Sequence[list[dict[str, Any]]],
    *,
    max_new_tokens: int,
) -> list[str]:
    # Qwen3-VL is decoder-only. Batched prompts can have different token lengths
    # (especially after inserting the generated attribute), so generation must
    # continue from the right edge of each prompt rather than from right-padding.
    tokenizer = getattr(target.processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    messages_batch = [[{"role": "user", "content": content}] for content in contents]
    rendered = [
        target.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for messages in messages_batch
    ]
    images: list[Any] = []
    videos: list[Any] = []
    for messages in messages_batch:
        batch_images, batch_videos = target.process_vision_info(messages)
        images.extend(batch_images or [])
        videos.extend(batch_videos or [])
    inputs = target.processor(
        text=rendered,
        images=images or None,
        videos=videos or None,
        padding=True,
        return_tensors="pt",
    )
    device = next(target.model.parameters()).device
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_width = int(inputs["input_ids"].shape[1])
    with target.torch.inference_mode():
        generated = target.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return target.processor.batch_decode(
        generated[:, prompt_width:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def generate_counterfactuals(
    config: Mapping[str, Any], output: Path, *, batch_size: int
) -> dict[str, Any]:
    _, test = load_safe_train_and_test(Path(str(config["manifest"])))
    cache_path = output / "counterfactual_descriptions_label_free.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(cache_path) if cache_path.is_file() else []
    by_id = {str(row["query_id"]): row for row in existing}
    pending = [sample for sample in test if sample.id not in by_id]
    target = build_target(dict(config["target_model"]))
    try:
        for begin in range(0, len(pending), batch_size):
            batch = pending[begin : begin + batch_size]
            attribute_contents = [
                [
                    target._image_block(sample.image),
                    {
                        "type": "text",
                        "text": (
                            "Identify the single visible attribute of this image most "
                            "relevant to deciding whether the bird is a landbird or a "
                            "waterbird. Return one short phrase under '### Attributes'."
                        ),
                    },
                ]
                for sample in batch
            ]
            attribute_raw = _batch_generate(
                target, attribute_contents, max_new_tokens=24
            )
            attributes = [parse_attribute(value) for value in attribute_raw]
            caption_contents = [
                [
                    target._image_block(sample.image),
                    {
                        "type": "text",
                        "text": (
                            "Write a concise target-image description that changes the "
                            f"visible attribute '{attribute}' to a different plausible "
                            "value while retaining the other image content. Return only "
                            "the modified description, no reasoning, at most 77 tokens."
                        ),
                    },
                ]
                for sample, attribute in zip(batch, attributes, strict=True)
            ]
            caption_raw = _batch_generate(
                target, caption_contents, max_new_tokens=72
            )
            rows = []
            for sample, raw_attribute, attribute, raw_caption in zip(
                batch, attribute_raw, attributes, caption_raw, strict=True
            ):
                caption = parse_caption(raw_caption)
                row = {
                    "query_id": sample.id,
                    "attribute": attribute,
                    "modified_caption": caption,
                    "attribute_response": raw_attribute,
                    "query_label_group_or_spurious_metadata_used": False,
                }
                by_id[sample.id] = row
                rows.append(row)
            append_jsonl(cache_path, rows)
            print(
                f"CIRCLES descriptions {len(by_id)}/{len(test)}",
                flush=True,
            )
    finally:
        del target
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    if len(by_id) != len(test):
        raise ValueError("Counterfactual-description cache is incomplete")
    result = {
        "queries": len(test),
        "complete": True,
        "cache_sha256": file_sha256(cache_path),
        "empty_attribute_count": sum(not str(row["attribute"]) for row in by_id.values()),
        "empty_caption_count": sum(
            not str(row["modified_caption"]) for row in by_id.values()
        ),
    }
    write_json(output / "description_status.json", result)
    return result


def _load_image_embeddings(
    source: Path, train: Sequence[Sample], test: Sequence[Sample]
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(source / "siglip2_train_validation_embeddings.npz", allow_pickle=False) as bundle:
        ids = [str(value) for value in bundle["sample_ids"]]
        values = normalize_rows(np.asarray(bundle["values"], dtype=np.float32))
    by_id = {value: index for index, value in enumerate(ids)}
    train_embeddings = np.stack([values[by_id[sample.id]] for sample in train])
    with np.load(source / "siglip2_test_embeddings.npz", allow_pickle=False) as bundle:
        ids = [str(value) for value in bundle["sample_ids"]]
        values = normalize_rows(np.asarray(bundle["values"], dtype=np.float32))
    by_id = {value: index for index, value in enumerate(ids)}
    test_embeddings = np.stack([values[by_id[sample.id]] for sample in test])
    return train_embeddings, test_embeddings


def select_stage(
    config: Mapping[str, Any], source: Path, output: Path, shots: Sequence[int]
) -> dict[str, Any]:
    train, test = load_safe_train_and_test(Path(str(config["manifest"])))
    descriptions = {
        str(row["query_id"]): row
        for row in load_jsonl(output / "counterfactual_descriptions_label_free.jsonl")
    }
    if set(descriptions) != {sample.id for sample in test}:
        raise ValueError("Counterfactual descriptions do not exactly cover test")
    train_embeddings, test_embeddings = _load_image_embeddings(source, train, test)
    retriever = build_retriever(dict(config["similarity_model"]))
    try:
        caption_embeddings = normalize_rows(
            retriever.embed_text(
                [descriptions[sample.id]["modified_caption"] for sample in test]
            ).astype(np.float32)
        )
    finally:
        del retriever
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    train_ids = np.asarray([sample.id for sample in train], dtype=str)
    zero = np.zeros(len(train), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(test):
        image_similarity = train_embeddings @ test_embeddings[query_index]
        composed_similarity = train_embeddings @ caption_embeddings[query_index]
        for shot in shots:
            original_count, composed_count = circles_budget(int(shot))
            originals = top_indices(
                image_similarity, zero, train_ids, k=original_count, lambda_value=0.0
            )
            composed = top_indices(
                composed_similarity, zero, train_ids, k=composed_count, lambda_value=0.0
            )
            # `top_indices` already follows the released RICES/CIRCLES convention:
            # highest-similarity example first, with deterministic id tie-breaking.
            original_order = list(map(int, originals))
            composed_order = list(map(int, composed))
            demo_indices = [*original_order, *composed_order]
            rows.append(
                {
                    "query_id": query.id,
                    "gold": "",
                    "lambda": 1.0,
                    "shots": int(shot),
                    "demos": [
                        {
                            "id": train[index].id,
                            "label": train[index].label,
                            "source": (
                                "original_image_retrieval"
                                if local_index < len(original_order)
                                else "counterfactual_composed_retrieval"
                            ),
                            "similarity": float(
                                image_similarity[index]
                                if local_index < len(original_order)
                                else composed_similarity[index]
                            ),
                        }
                        for local_index, index in enumerate(demo_indices)
                    ],
                    "attribute": descriptions[query.id]["attribute"],
                    "modified_caption": descriptions[query.id]["modified_caption"],
                    "original_demo_count": original_count,
                    "composed_demo_count": composed_count,
                    "prompt_order": CIRCLES_PROMPT_ORDER,
                    "query_label_group_or_spurious_metadata_used": False,
                }
            )
    selections_path = output / "selections_label_free.jsonl"
    write_jsonl(selections_path, rows)
    protocol = {
        "status": "frozen_before_circles_inference",
        "dataset": "Waterbirds",
        "split": "test",
        "query_count": len(test),
        "shots": list(map(int, shots)),
        "method": "equal-total-budget CIRCLES retrieval adaptation",
        "official_code": "https://github.com/gzxiong/CIRCLES",
        "official_default": "16 original image retrievals + 16 composed retrievals",
        "budget_rule": "ceil(K/2) original + floor(K/2) composed = K total images",
        "target_model": str(config["target_model"]["model"]),
        "similarity_model": str(config["similarity_model"]["model"]),
        "waterbirds_training_text_is_empty_so_official_task-text_term_is_constant": True,
        "official_list_concatenation_does_not_deduplicate": True,
        "adaptation_changes": [
            "same SigLIP2 retriever as RICES/ours instead of official CLIP ViT-g/14",
            "same Qwen3-VL-4B target as RICES/ours",
            "direct <=77-token target-caption generation instead of a long structured rationale",
            "caption headers are not injected into the target prompt so selection is compared under an identical prompt",
        ],
        "same_classification_prompt_as_rices_and_self_discovered": True,
        "prompt_order": CIRCLES_PROMPT_ORDER,
        "reference_rices_and_self_discovered_prompt_order": REFERENCE_PROMPT_ORDER,
        "prompt_order_matched_to_reference": False,
        "modified_caption_used_for_retrieval_but_not_shown_to_target_vlm": True,
        "test_labels_groups_or_spurious_metadata_used_for_selection": False,
        "test_already_consumed_by_prior_3shot_result": True,
        "input_sha256": {
            "config": file_sha256(Path(str(config["_config_path"]))),
            "manifest": file_sha256(Path(str(config["manifest"]))),
            "descriptions": file_sha256(
                output / "counterfactual_descriptions_label_free.jsonl"
            ),
            "train_validation_embeddings": file_sha256(
                source / "siglip2_train_validation_embeddings.npz"
            ),
            "test_embeddings": file_sha256(source / "siglip2_test_embeddings.npz"),
        },
        "selection_sha256": file_sha256(selections_path),
    }
    write_json(output / "protocol.json", protocol)
    return protocol


def _verify(config: Mapping[str, Any], source: Path, output: Path) -> dict[str, Any]:
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_circles_inference":
        raise ValueError("CIRCLES protocol is not frozen")
    checks = {
        "config": Path(str(config["_config_path"])),
        "manifest": Path(str(config["manifest"])),
        "descriptions": output / "counterfactual_descriptions_label_free.jsonl",
        "train_validation_embeddings": source / "siglip2_train_validation_embeddings.npz",
        "test_embeddings": source / "siglip2_test_embeddings.npz",
    }
    for name, path in checks.items():
        if file_sha256(path) != protocol["input_sha256"][name]:
            raise ValueError(f"Frozen CIRCLES input changed: {name}")
    if file_sha256(output / "selections_label_free.jsonl") != protocol["selection_sha256"]:
        raise ValueError("Frozen CIRCLES selections changed")
    return protocol


def infer_stage(
    config: Mapping[str, Any], source: Path, base: Path, output: Path
) -> dict[str, Any]:
    protocol = _verify(config, source, output)
    train, test = load_safe_train_and_test(Path(str(config["manifest"])))
    selections = load_jsonl(output / "selections_label_free.jsonl")
    cache_path = output / "context_predictions_cache.jsonl"
    if not cache_path.is_file():
        seed = base / "context_predictions_cache.jsonl"
        write_jsonl(cache_path, load_jsonl(seed) if seed.is_file() else [])
    predictions: list[dict[str, Any]] = []
    for shot in protocol["shots"]:
        shot_rows = [row for row in selections if int(row["shots"]) == int(shot)]
        frozen = predict_selection_rows(
            config,
            [*train, *test],
            shot_rows,
            selected_lambda=1.0,
            cache_path=cache_path,
        )
        for row in frozen:
            row["condition"] = "circles"
            row["shots"] = int(shot)
        predictions.extend(frozen)
        write_jsonl(output / "predictions_label_free.jsonl", predictions)
        print(f"completed CIRCLES {shot}-shot", flush=True)
    result = {
        "complete": True,
        "prediction_rows": len(predictions),
        "prediction_sha256": file_sha256(output / "predictions_label_free.jsonl"),
    }
    write_json(output / "inference_status.json", result)
    return result


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = [bool(row["correct"]) for row in rows]
    by_group: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group"])].append(bool(row["correct"]))
    group_accuracy = {
        key: float(np.mean(values)) for key, values in sorted(by_group.items())
    }
    return {
        "count": len(rows),
        "accuracy": float(np.mean(correct)),
        "group_accuracy": group_accuracy,
        "worst_group_accuracy": min(group_accuracy.values()),
    }


def report_stage(
    config: Mapping[str, Any], source: Path, base: Path, output: Path
) -> dict[str, Any]:
    protocol = _verify(config, source, output)
    predictions = load_jsonl(output / "predictions_label_free.jsonl")
    if len(predictions) != int(protocol["query_count"]) * len(protocol["shots"]):
        raise ValueError("CIRCLES predictions are incomplete")
    samples = load_manifest(config["manifest"])
    sample_by_id = {sample.id: sample for sample in samples}
    selections = load_jsonl(output / "selections_label_free.jsonl")
    selection_by_key = {
        (str(row["query_id"]), int(row["shots"])): row for row in selections
    }
    base_summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    by_shot: dict[str, Any] = {}
    scored_rows: list[dict[str, Any]] = []
    for shot in protocol["shots"]:
        cell_rows = []
        for row in predictions:
            if int(row["shots"]) != int(shot):
                continue
            query = sample_by_id[str(row["query_id"])]
            scored = dict(row) | {
                "gold": query.label,
                "group": query.group,
                "correct": str(row["prediction"]) == query.label,
            }
            cell_rows.append(scored)
            scored_rows.append(scored)
        metrics = _metrics(cell_rows)
        selected = [
            selection_by_key[(str(row["query_id"]), int(shot))]
            for row in cell_rows
        ]
        flags = [
            bool(sample_by_id[str(demo["id"])].metadata["diagnostic"])
            for row in selected
            for demo in row["demos"]
        ]
        metrics["retrieval_audit"] = {
            "demo_slots": len(flags),
            "diagnostic_demo_rate": float(np.mean(flags)),
            "contexts_with_diagnostic_demo": sum(
                any(
                    bool(sample_by_id[str(demo["id"])].metadata["diagnostic"])
                    for demo in row["demos"]
                )
                for row in selected
            ),
        }
        base_cell = base_summary["by_shot"][str(shot)]
        circles_by_id = {str(row["query_id"]): row for row in cell_rows}
        rices_rows = [
            row
            for row in load_jsonl(base / "predictions_scored_posthoc.jsonl")
            if int(row["shots"]) == int(shot) and row["condition"] == "rices"
        ]
        wins = sum(
            not bool(row["correct"])
            and bool(circles_by_id[str(row["query_id"])]["correct"])
            for row in rices_rows
        )
        losses = sum(
            bool(row["correct"])
            and not bool(circles_by_id[str(row["query_id"])]["correct"])
            for row in rices_rows
        )
        paired_rows = [
            {
                "query_id": str(row["query_id"]),
                "condition": "rices",
                "correct": bool(row["correct"]),
            }
            for row in rices_rows
        ] + [
            {
                "query_id": str(row["query_id"]),
                "condition": "circles",
                "correct": bool(row["correct"]),
            }
            for row in cell_rows
        ]
        by_shot[str(shot)] = {
            "rices": base_cell["rices"],
            "circles": metrics,
            "self_discovered": base_cell["self_discovered"],
            "gain_vs_rices": {
                "circles_accuracy": metrics["accuracy"] - base_cell["rices"]["accuracy"],
                "circles_worst_group": metrics["worst_group_accuracy"]
                - base_cell["rices"]["worst_group_accuracy"],
                "self_discovered_accuracy": base_cell["gain"]["accuracy"],
                "self_discovered_worst_group": base_cell["gain"]["worst_group_accuracy"],
            },
            "circles_paired_vs_rices": {"wins": wins, "losses": losses},
            "circles_paired_inference": paired_delta_statistics(
                paired_rows,
                method="circles",
                seed=20260828 + int(shot),
            ),
        }
    write_jsonl(output / "predictions_scored_posthoc.jsonl", scored_rows)
    summary = {
        "protocol": protocol,
        "by_shot": by_shot,
        "query_metadata_joined_only_after_all_predictions_completed": True,
        "comparison_note": (
            "CIRCLES is adapted to Waterbirds and an equal total image budget; "
            "this is not an exact reproduction of its supported-dataset protocol. "
            "The saved CIRCLES contexts use official highest-first ordering, while "
            "the referenced RICES/self-discovered contexts place the most similar "
            "demo nearest the query. Consequently, cross-method deltas are "
            "order-confounded and must not be presented as a fully controlled "
            "head-to-head comparison until an order-matched rerun is complete."
        ),
        "prompt_order_matched_to_reference": bool(
            protocol.get("prompt_order_matched_to_reference", False)
        ),
    }
    write_json(output / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CIRCLES-style Waterbirds full-test baseline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage", choices=("describe", "select", "infer", "report", "all"), default="all"
    )
    parser.add_argument("--shots", default="1,2,3,4,8")
    parser.add_argument("--generation-batch-size", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.stage in {"describe", "all"}:
        result = generate_counterfactuals(
            config, args.output, batch_size=args.generation_batch_size
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"select", "all"}:
        result = select_stage(
            config, args.source, args.output, parse_shots(args.shots)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"infer", "all"}:
        result = infer_stage(config, args.source, args.base, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"report", "all"}:
        result = report_stage(config, args.source, args.base, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

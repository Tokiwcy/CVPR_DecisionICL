from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .hashing import stable_hash
from .schema import AtomState, FactorGroup


DEFAULT_PERSPECTIVES = (
    "part shape and relative proportions",
    "color, markings, material, and texture",
    "pose, viewpoint, silhouette, and spatial structure",
    "fine-grained contrasts between easily confused classes",
    "negative evidence: visible cues that argue against a plausible class",
    "context cues the model actually uses, including potentially spurious background cues",
)


class DiscoveryBackend(Protocol):
    model_id: str

    def generate(self, image_paths: Sequence[str], prompt: str) -> str: ...


@dataclass
class TransformersDiscoveryBackend:
    model_id: str
    model_revision: str = "local"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 1536
    min_pixels: int | None = 65_536
    max_pixels: int | None = 401_408

    def __post_init__(self) -> None:
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            self.dtype
        ]
        local = Path(self.model_id).exists()
        revision = None if self.model_revision == "local" else self.model_revision
        processor_kwargs: dict[str, int] = {}
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=revision,
            local_files_only=local,
            **processor_kwargs,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            revision=revision,
            local_files_only=local,
            dtype=dtype,
        ).eval().to(self.device)

    def generate(self, image_paths: Sequence[str], prompt: str) -> str:
        import torch
        from PIL import Image

        self._load()
        assert self._model is not None and self._processor is not None
        content: list[dict[str, str]] = [{"type": "image", "image": path} for path in image_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        rendered = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = self._processor(text=[rendered], images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        raw = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        if time.perf_counter() - started <= 0:
            raise RuntimeError("invalid discovery runtime clock")
        return raw.strip()


@dataclass(frozen=True)
class DiscoveryRequest:
    prompt_id: str
    batch_id: str
    sample_ids: tuple[str, ...]
    image_paths: tuple[str, ...]
    perspective: str
    prompt: str


def build_discovery_prompt(
    *,
    task_instruction: str,
    class_names: Sequence[str],
    perspective: str,
    max_groups: int,
) -> str:
    classes = ", ".join(class_names)
    return f"""You are the target vision-language model whose decision factors we are eliciting.

Task: {task_instruction}
Allowed classes: {classes}
Perspective for this pass: {perspective}

Inspect all supplied TRAIN images. Propose at most {max_groups} visible factor groups that you would use to distinguish the allowed classes. Seek a different axis from the other perspectives. A factor must be reusable across most images in this task, visually measurable from pixels, and stated without referring to a class name or sample ID. Do not merely list salient objects.

Each factor is one well-defined visual question with 2--6 mutually comparable states. Prefer contrastive multi-state axes (for example short/medium/long) over isolated yes/no predicates. For an ordered axis, assign distinct numeric coordinates increasing with the named property. Add an abstain state only when the visual property can genuinely be unobservable; an abstain state has no coordinate. State descriptions must be complete image descriptions suitable for a frozen image-text encoder. Do not merge unrelated attributes into one factor.

Return JSON only, with exactly this shape:
{{
  "groups": [
    {{
      "region": "beak",
      "attribute": "length",
      "question": "How long is the visible bird beak relative to the head?",
      "applies_to": "images in which the bird beak is visible",
      "states": [
        {{"label": "short", "visual_description": "a bird with a short beak relative to its head", "coordinate": -1}},
        {{"label": "medium", "visual_description": "a bird with a medium-length beak relative to its head", "coordinate": 0}},
        {{"label": "long", "visual_description": "a bird with a long beak relative to its head", "coordinate": 1}}
      ]
    }}
  ]
}}
"""


def make_discovery_requests(
    *,
    sample_ids: Sequence[str],
    image_paths: Sequence[str],
    task_instruction: str,
    class_names: Sequence[str],
    batch_size: int = 6,
    max_groups_per_perspective: int = 6,
    perspectives: Sequence[str] = DEFAULT_PERSPECTIVES,
) -> list[DiscoveryRequest]:
    if len(sample_ids) != len(image_paths):
        raise ValueError("sample_ids and image_paths must have the same length")
    if batch_size < 2:
        raise ValueError("discovery batch_size must be at least two")
    requests: list[DiscoveryRequest] = []
    for start in range(0, len(sample_ids), batch_size):
        batch_ids = tuple(sample_ids[start : start + batch_size])
        batch_paths = tuple(image_paths[start : start + batch_size])
        if len(batch_ids) < 2:
            continue
        batch_id = f"batch-{start // batch_size:04d}"
        for perspective_index, perspective in enumerate(perspectives):
            prompt = build_discovery_prompt(
                task_instruction=task_instruction,
                class_names=class_names,
                perspective=perspective,
                max_groups=max_groups_per_perspective,
            )
            prompt_id = stable_hash(
                {
                    "batch_id": batch_id,
                    "sample_ids": batch_ids,
                    "perspective_index": perspective_index,
                    "prompt": prompt,
                }
            )[:16]
            requests.append(
                DiscoveryRequest(
                    prompt_id=prompt_id,
                    batch_id=batch_id,
                    sample_ids=batch_ids,
                    image_paths=batch_paths,
                    perspective=perspective,
                    prompt=prompt,
                )
            )
    return requests


def _identifier(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized or "unnamed"


def _extract_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        left = cleaned.find("{")
        right = cleaned.rfind("}")
        if left < 0 or right <= left:
            raise ValueError("discovery response does not contain a JSON object")
        parsed = json.loads(cleaned[left : right + 1])
    if not isinstance(parsed, dict):
        raise ValueError("discovery response root must be an object")
    return parsed


def parse_discovery_response(
    raw: str,
    *,
    request: DiscoveryRequest,
    source_model: str,
) -> tuple[FactorGroup, ...]:
    parsed = _extract_json(raw)
    raw_groups = parsed.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError("discovery response must contain a groups list")
    groups: list[FactorGroup] = []
    for group_index, item in enumerate(raw_groups):
        if not isinstance(item, dict):
            raise ValueError(f"group {group_index}: expected an object")
        region = str(item.get("region", "")).strip()
        attribute = str(item.get("attribute", "")).strip()
        raw_states = item.get("states")
        if not isinstance(raw_states, list):
            raise ValueError(f"group {group_index}: states must be a list")
        states: list[AtomState] = []
        for state_index, state in enumerate(raw_states):
            if not isinstance(state, dict):
                raise ValueError(f"group {group_index}, state {state_index}: expected an object")
            label = str(state.get("label", "")).strip()
            coordinate = state.get("coordinate")
            abstain = bool(state.get("abstain", False))
            if coordinate is not None:
                coordinate = float(coordinate)
            states.append(
                AtomState(
                    state_id=f"{_identifier(label)}-{state_index}",
                    label=label,
                    visual_description=str(state.get("visual_description", "")).strip(),
                    coordinate=coordinate,
                    abstain=abstain,
                )
            )
        group_id = (
            f"{_identifier(region)}--{_identifier(attribute)}--"
            f"{request.batch_id}--{request.prompt_id[:8]}--{group_index:02d}"
        )
        group = FactorGroup(
            group_id=group_id,
            region=region,
            attribute=attribute,
            question=str(item.get("question", "")).strip(),
            applies_to=str(item.get("applies_to", "")).strip(),
            states=tuple(states),
            perspective=request.perspective,
            source_model=source_model,
            source_prompt_id=request.prompt_id,
            discovery_batch_id=request.batch_id,
            source_sample_ids=request.sample_ids,
        )
        group.validate()
        groups.append(group)
    return tuple(groups)

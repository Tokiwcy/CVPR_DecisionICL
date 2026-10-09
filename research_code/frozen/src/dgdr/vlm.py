from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

from .hashing import file_sha256, stable_hash
from .plans import RetrievalPlan
from .schema import Sample


@dataclass(frozen=True)
class ClassificationPrompt:
    task_instruction: str
    class_to_code: dict[str, str]
    version: str = "classification-codes-v1"

    def validate(self) -> None:
        if len(self.class_to_code) < 2:
            raise ValueError("at least two classes are required")
        if len(set(self.class_to_code.values())) != len(self.class_to_code):
            raise ValueError("class codes must be unique")
        for code in self.class_to_code.values():
            if not re.fullmatch(r"C[0-9]{3,}", code):
                raise ValueError(f"invalid class code: {code}")


def make_classification_prompt(task_instruction: str, class_names: Sequence[str]) -> ClassificationPrompt:
    mapping = {name: f"C{index:03d}" for index, name in enumerate(sorted(class_names))}
    prompt = ClassificationPrompt(task_instruction=task_instruction, class_to_code=mapping)
    prompt.validate()
    return prompt


class ClassificationBackend(Protocol):
    model_id: str
    model_revision: str

    def classify(
        self,
        *,
        query: Sample,
        demos: Sequence[Sample],
        prompt: ClassificationPrompt,
    ) -> tuple[str, str, dict[str, float | int | str]]: ...


@dataclass
class TransformersChatVLM:
    model_id: str
    model_revision: str = "local"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 8
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
        processor_kwargs = {}
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        local = Path(self.model_id).exists()
        revision = None if self.model_revision == "local" else self.model_revision
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

    def classify(
        self,
        *,
        query: Sample,
        demos: Sequence[Sample],
        prompt: ClassificationPrompt,
    ) -> tuple[str, str, dict[str, float | int | str]]:
        import torch
        from PIL import Image

        self._load()
        assert self._model is not None and self._processor is not None
        prompt.validate()
        legend = "\n".join(f"{code}: {label}" for label, code in prompt.class_to_code.items())
        content: list[dict[str, str]] = [
            {
                "type": "text",
                "text": f"Task: {prompt.task_instruction}\nAllowed outputs:\n{legend}\nStudy the demonstrations.",
            }
        ]
        image_paths: list[str] = []
        for index, demo in enumerate(demos, start=1):
            content.append({"type": "image", "image": demo.image_path})
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Demonstration {index}. Question: {demo.question or prompt.task_instruction}\n"
                        f"Answer: {prompt.class_to_code[demo.label]}"
                    ),
                }
            )
            image_paths.append(demo.image_path)
        content.append({"type": "image", "image": query.image_path})
        content.append(
            {
                "type": "text",
                "text": (
                    f"Query. Question: {query.question or prompt.task_instruction}\n"
                    "Return exactly one class code and nothing else."
                ),
            }
        )
        image_paths.append(query.image_path)
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
        prompt_length = inputs["input_ids"].shape[1]
        generated = output[:, prompt_length:]
        raw = self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        elapsed = time.perf_counter() - started
        matches = [code for code in prompt.class_to_code.values() if re.search(rf"(?<!\w){re.escape(code)}(?!\w)", raw)]
        prediction = matches[0] if len(matches) == 1 else "PARSE_ERROR"
        return prediction, raw, {"elapsed_seconds": elapsed, "generated_tokens": int(generated.shape[1])}


def _sample_hash(sample: Sample) -> str:
    return sample.image_sha256 or file_sha256(sample.image_path)


def inference_request_hash(
    *,
    backend: ClassificationBackend,
    query: Sample,
    demos: Sequence[Sample],
    prompt: ClassificationPrompt,
) -> str:
    return stable_hash(
        {
            "model_id": backend.model_id,
            "model_revision": backend.model_revision,
            "backend_class": f"{type(backend).__module__}.{type(backend).__qualname__}",
            "inference_settings": {
                name: getattr(backend, name)
                for name in ("dtype", "min_pixels", "max_pixels", "max_new_tokens")
                if hasattr(backend, name)
            },
            "prompt": {
                "task_instruction": prompt.task_instruction,
                "class_to_code": prompt.class_to_code,
                "version": prompt.version,
            },
            "query": {
                "sample_id": query.sample_id,
                "image_sha256": _sample_hash(query),
                "question": query.question,
            },
            "demos": [
                {
                    "sample_id": demo.sample_id,
                    "image_sha256": _sample_hash(demo),
                    "label": demo.label,
                    "question": demo.question,
                }
                for demo in demos
            ],
        }
    )


def run_inference_plan(
    *,
    plan: RetrievalPlan,
    queries: Sequence[Sample],
    candidates: Sequence[Sample],
    prompt: ClassificationPrompt,
    backend: ClassificationBackend,
    cache: dict[str, dict[str, object]] | None = None,
    checkpoint: Callable[[dict[str, dict[str, object]]], None] | None = None,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    plan.validate(queries, candidates)
    by_candidate_id = {sample.sample_id: sample for sample in candidates}
    cache_values = {} if cache is None else dict(cache)
    rows: list[dict[str, object]] = []
    for query, plan_row in zip(queries, plan.rows, strict=True):
        demos = [by_candidate_id[sample_id] for sample_id in plan_row["demo_ids"]]
        request_hash = inference_request_hash(backend=backend, query=query, demos=demos, prompt=prompt)
        if request_hash in cache_values:
            cached = cache_values[request_hash]
            code = str(cached["prediction_code"])
            raw = str(cached["raw_output"])
            runtime = dict(cached.get("runtime", {}))
            reused = True
        else:
            code, raw, runtime = backend.classify(query=query, demos=demos, prompt=prompt)
            cache_values[request_hash] = {
                "prediction_code": code,
                "raw_output": raw,
                "runtime": runtime,
            }
            if checkpoint is not None:
                checkpoint(cache_values)
            reused = False
        inverse = {code: label for label, code in prompt.class_to_code.items()}
        rows.append(
            {
                "query_id": query.sample_id,
                "gold_label": query.label,
                "prediction_code": code,
                "prediction_label": inverse.get(code, "PARSE_ERROR"),
                "raw_output": raw,
                "request_hash": request_hash,
                "cache_reused": reused,
                "runtime": runtime,
                "method": plan.method,
            }
        )
    return rows, cache_values

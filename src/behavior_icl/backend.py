from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object from a model response."""
    decoder = json.JSONDecoder()
    for position, value in enumerate(text):
        if value != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Response does not contain a valid JSON object")


class QwenVLBackend:
    """Minimal local Qwen2.5-VL/Qwen3-VL backend used by the released experiments."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda",
        max_pixels: int = 128 * 28 * 28,
        min_pixels: int = 32 * 28 * 28,
        load_in_4bit: bool = False,
        max_cached_images: int = 8192,
    ) -> None:
        import torch
        from transformers import (
            AutoConfig,
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
        )

        self.torch = torch
        self.device = device
        self.model_path = str(Path(model_path).resolve())
        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        self.model_type = str(getattr(config, "model_type", ""))
        self.max_cached_images = max(0, max_cached_images)
        self._image_cache: OrderedDict[str, Image.Image] = OrderedDict()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            if hasattr(image_processor, "max_pixels"):
                image_processor.max_pixels = max_pixels
            if hasattr(image_processor, "min_pixels"):
                image_processor.min_pixels = min_pixels

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "dtype": dtype,
            "attn_implementation": "sdpa",
        }
        if load_in_4bit:
            if not device.startswith("cuda"):
                raise ValueError("4-bit loading requires CUDA")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        model_class = {
            "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
            "qwen3_vl": Qwen3VLForConditionalGeneration,
        }.get(self.model_type)
        if model_class is None:
            raise ValueError(f"Unsupported Qwen-VL model type: {self.model_type!r}")
        self.model = model_class.from_pretrained(self.model_path, **model_kwargs)
        if not load_in_4bit:
            self.model.to(device)
        self.model.eval()

    def _open_image(self, path: str | Path) -> Image.Image:
        key = str(Path(path).resolve())
        cached = self._image_cache.get(key)
        if cached is not None:
            self._image_cache.move_to_end(key)
            return cached
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((448, 448))
        if self.max_cached_images:
            self._image_cache[key] = image
            self._image_cache.move_to_end(key)
            while len(self._image_cache) > self.max_cached_images:
                self._image_cache.popitem(last=False)
        return image

    def _to_device(self, inputs: Any) -> dict[str, Any]:
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def generate_json(
        self,
        image_paths: Iterable[str | Path],
        prompt: str,
        *,
        max_new_tokens: int = 256,
    ) -> tuple[dict[str, Any], str]:
        content: list[dict[str, Any]] = [
            {"type": "image", "image": self._open_image(path)}
            for path in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a precise visual analyst. Follow the requested JSON schema.",
                    }
                ],
            },
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self._to_device(inputs)
        budgets = [max_new_tokens]
        retry_budget = max(640, max_new_tokens * 2)
        if retry_budget > max_new_tokens:
            budgets.append(retry_budget)
        last_text = ""
        last_error: ValueError | None = None
        for budget in budgets:
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=budget,
                    do_sample=False,
                    use_cache=True,
                )
            prompt_length = inputs["input_ids"].shape[1]
            last_text = self.processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            try:
                return extract_json_object(last_text), last_text
            except ValueError as error:
                last_error = error
        raise ValueError(
            f"Could not parse JSON after token budgets {budgets}; response starts {last_text[:300]!r}"
        ) from last_error

    def score_named_classes_batch(
        self,
        requests: list[tuple[list[dict[str, Any]], dict[str, Any]]],
        label_names: list[str],
        label_codes: list[str],
        *,
        task_instruction: str,
    ) -> list[dict[str, Any]]:
        if not requests:
            return []
        if len(label_names) != len(label_codes) or not label_names:
            raise ValueError("label_names and label_codes must be non-empty and aligned")
        code_ids: list[int] = []
        for code in label_codes:
            encoded = self.processor.tokenizer.encode(code, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Class code {code!r} is not one token: {encoded}")
            code_ids.append(encoded[0])
        if len(set(code_ids)) != len(code_ids):
            raise ValueError("Class codes do not map to unique token ids")
        code_by_label = dict(zip(label_names, label_codes, strict=True))
        catalog = "\n".join(
            f"{code}: {label}"
            for label, code in zip(label_names, label_codes, strict=True)
        )

        def question(example: dict[str, Any]) -> str:
            text = str(example.get("text", "")).strip()
            prefix = f"{task_instruction}\n"
            if text:
                prefix += f"Additional input:\n{text}\n"
            reasoning_rules = str(example.get("reasoning_rules", "")).strip()
            if reasoning_rules:
                prefix += (
                    "Validated decision rules:\n"
                    f"{reasoning_rules}\n"
                    "Inspect the listed cues before choosing. Ignore cues that are not visible, "
                    "and do not let background alone override the classified subject.\n"
                )
            return prefix + "Return exactly one class code and no other text."

        conversations: list[list[dict[str, Any]]] = []
        for demonstrations, query in requests:
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Classify visual inputs using the fixed catalog below. "
                                "Codes are output identifiers; reason from images, class names, "
                                "and demonstrations.\n\nCLASS CATALOG\n" + catalog
                            ),
                        }
                    ],
                }
            ]
            for example in demonstrations:
                label = str(example["label"])
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": self._open_image(example["path"])},
                            {"type": "text", "text": question(example)},
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": code_by_label[label]}],
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self._open_image(query["path"])},
                        {"type": "text", "text": question(query)},
                    ],
                }
            )
            conversations.append(messages)

        previous_padding = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        try:
            inputs = self.processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
            )
        finally:
            self.processor.tokenizer.padding_side = previous_padding
        inputs = self._to_device(inputs)
        with self.torch.inference_mode():
            try:
                output = self.model(**inputs, use_cache=False, logits_to_keep=1)
            except TypeError:
                output = self.model(**inputs, use_cache=False)
            logits = output.logits[:, -1].float()[:, code_ids]
            probabilities = self.torch.softmax(logits, dim=1).cpu().numpy()
        results: list[dict[str, Any]] = []
        for row in probabilities:
            index = int(np.argmax(row))
            results.append(
                {
                    "prediction": label_names[index],
                    "prediction_code": label_codes[index],
                    "probabilities": {
                        label: float(row[position])
                        for position, label in enumerate(label_names)
                    },
                }
            )
        return results

    def close(self) -> None:
        self._image_cache.clear()
        del self.model
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()

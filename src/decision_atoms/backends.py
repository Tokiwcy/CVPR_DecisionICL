from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile

from .math_utils import softmax
from .prompts import atom_prompt, class_prompt, discovery_prompt, icl_prompt, task_icl_content
from .types import DecisionAtom, Sample


class TargetVLM(ABC):
    @abstractmethod
    def class_scores(self, image: str, labels: list[str], prompt_variant: int = 0) -> dict[str, float]:
        raise NotImplementedError

    def class_scores_batch(
        self, images: list[str], labels: list[str], prompt_variant: int = 0
    ) -> list[dict[str, float]]:
        return [self.class_scores(image, labels, prompt_variant) for image in images]

    @abstractmethod
    def discover(
        self,
        images_a: list[str],
        images_b: list[str],
        label_a: str,
        label_b: str,
        source_mode: str,
        atoms_per_bootstrap: int,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def atom_probabilities(self, image: str, atom: DecisionAtom, prompt_variant: int) -> np.ndarray:
        """Return probabilities in A, B, U order."""
        raise NotImplementedError

    def atom_probabilities_batch(
        self, image: str, atoms: list[DecisionAtom], prompt_variant: int
    ) -> np.ndarray:
        return np.stack([self.atom_probabilities(image, atom, prompt_variant) for atom in atoms])

    @abstractmethod
    def icl_scores(self, query: Sample, demos: list[Sample], labels: list[str]) -> dict[str, float]:
        raise NotImplementedError

    def icl_scores_batch(
        self, queries: list[Sample], demos: list[list[Sample]], labels: list[str]
    ) -> list[dict[str, float]]:
        return [self.icl_scores(query, rows, labels) for query, rows in zip(queries, demos)]

    def task_scores(self, query: Sample, demos: list[Sample], labels: list[str]) -> dict[str, float]:
        """Score a query whose candidate set may be global or sample-specific."""
        if query.options:
            raise NotImplementedError("this backend does not support per-sample candidate sets")
        return self.icl_scores(query, demos, labels)

    def task_scores_batch(
        self, queries: list[Sample], demos: list[list[Sample]], labels: list[str]
    ) -> list[dict[str, float]]:
        return [self.task_scores(query, rows, labels) for query, rows in zip(queries, demos)]


class VisualRetriever(ABC):
    @abstractmethod
    def embed(self, images: list[str]) -> np.ndarray:
        raise NotImplementedError


def _open_rgb_image(path: str) -> Image.Image:
    """Load one image, tolerating only Pillow's explicit truncated-file error."""
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except OSError as error:
        if "truncated" not in str(error).lower():
            raise
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous


class MockTargetVLM(TargetVLM):
    """Deterministic image-dependent backend for tests; it never reads labels from a manifest."""

    @staticmethod
    def _features(image: str) -> np.ndarray:
        pixels = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32) / 255.0
        return np.asarray([*pixels.mean(axis=(0, 1)), pixels.std()], dtype=np.float32)

    def class_scores(self, image: str, labels: list[str], prompt_variant: int = 0) -> dict[str, float]:
        features = self._features(image)
        red_blue = float(features[0] - features[2])
        offset = (prompt_variant - 1) * 0.005
        scores: dict[str, float] = {}
        for index, label in enumerate(labels):
            lowered = label.lower()
            if "red" in lowered or "landbird" in lowered:
                score = red_blue
            elif "blue" in lowered or "waterbird" in lowered:
                score = -red_blue
            else:
                score = -abs(index / max(len(labels) - 1, 1) - features[1])
            scores[label] = float(score + offset * (index + 1))
        return scores

    def discover(
        self,
        images_a: list[str],
        images_b: list[str],
        label_a: str,
        label_b: str,
        source_mode: str,
        atoms_per_bootstrap: int,
    ) -> str:
        payload = [
            {"concept": "dominant color", "a_side": "warmer or redder", "b_side": "cooler or bluer"},
            {"concept": "brightness", "a_side": "brighter", "b_side": "darker"},
        ][:atoms_per_bootstrap]
        return json.dumps(payload)

    def atom_probabilities(self, image: str, atom: DecisionAtom, prompt_variant: int) -> np.ndarray:
        features = self._features(image)
        if "color" in atom.concept.lower():
            evidence = 5.0 * float(features[0] - features[2])
        elif "bright" in atom.concept.lower():
            evidence = 4.0 * float(features[:3].mean() - 0.5)
        else:
            evidence = 0.0
        if "blue" in atom.a_side.lower() or "red" in atom.b_side.lower():
            evidence = -evidence
        uncertainty = 0.5 - abs(evidence)
        return softmax(np.asarray([evidence, -evidence, uncertainty + prompt_variant * 0.001]))

    def icl_scores(self, query: Sample, demos: list[Sample], labels: list[str]) -> dict[str, float]:
        return self.class_scores(query.image, labels)


class MockRetriever(VisualRetriever):
    def embed(self, images: list[str]) -> np.ndarray:
        return np.stack([MockTargetVLM._features(image) for image in images])


class OpenAIResponsesBackend(TargetVLM):
    """Vision target backed by Responses API output-token log probabilities."""

    def __init__(self, config: dict[str, Any]):
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("OpenAI backend requires the official `openai` package") from error
        api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing OpenAI API key environment variable: {api_key_env}")
        self.client = OpenAI(
            api_key=api_key,
            timeout=float(config.get("timeout", 120.0)),
            max_retries=int(config.get("max_retries", 5)),
        )
        self.model = str(config["model"])
        self.detail = str(config.get("detail", "high"))
        self.reasoning_effort = str(config.get("reasoning_effort", "none"))
        self.top_logprobs = int(config.get("top_logprobs", 20))
        self.score_mode = str(config.get("score_mode", "logprobs"))
        self.max_output_tokens = int(config.get("max_output_tokens", 16))
        self.workers = int(config.get("workers", 8))
        self.max_api_calls = int(config.get("max_api_calls", 10_000))
        self.cache_dir = Path(config.get("cache_dir", ".openai_response_cache")).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._call_count = 0
        self._lock = threading.Lock()

    def _image_item(self, image: str) -> dict[str, str]:
        path = Path(image).resolve()
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "input_image",
            "image_url": f"data:{mime_type};base64,{payload}",
            "detail": self.detail,
        }

    @staticmethod
    def _normalized_token(token: str) -> str:
        return token.strip().strip("`*.,:;()[]{}")

    def _extract_scores(self, payload: dict[str, Any], candidates: list[str]) -> dict[str, float]:
        text = ""
        positions: list[dict[str, Any]] = []
        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") != "output_text":
                    continue
                text += str(content.get("text", ""))
                positions.extend(content.get("logprobs") or [])
        candidate_set = set(candidates)
        chosen_position = None
        for position in positions:
            observed = self._normalized_token(str(position.get("token", "")))
            alternatives = {
                self._normalized_token(str(row.get("token", "")))
                for row in position.get("top_logprobs", [])
            }
            if observed in candidate_set or alternatives & candidate_set:
                chosen_position = position
                break
        if chosen_position is None:
            raise RuntimeError(
                f"OpenAI response has no candidate-code logprobs; text={text!r}, candidates={candidates}"
            )
        score_map: dict[str, float] = {}
        rows = [chosen_position, *(chosen_position.get("top_logprobs") or [])]
        for row in rows:
            token = self._normalized_token(str(row.get("token", "")))
            if token in candidate_set:
                score_map[token] = max(score_map.get(token, -np.inf), float(row["logprob"]))
        missing = [candidate for candidate in candidates if candidate not in score_map]
        if missing:
            raise RuntimeError(
                f"candidate codes absent from top-{self.top_logprobs}: {missing}; text={text!r}"
            )
        return score_map

    def _extract_chat_scores(
        self, payload: dict[str, Any], candidates: list[str]
    ) -> dict[str, float]:
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI Chat Completions response contains no choices")
        positions = ((choices[0].get("logprobs") or {}).get("content") or [])
        candidate_set = set(candidates)
        for position in positions:
            rows = list(position.get("top_logprobs") or [])
            observed_token = str(position.get("token", ""))
            if not any(str(row.get("token", "")) == observed_token for row in rows):
                rows.append(position)
            score_map: dict[str, float] = {}
            for row in rows:
                token = self._normalized_token(str(row.get("token", "")))
                if token in candidate_set:
                    value = float(row["logprob"])
                    score_map[token] = float(np.logaddexp(score_map.get(token, -np.inf), value))
            if all(candidate in score_map for candidate in candidates):
                return score_map
        text = str((choices[0].get("message") or {}).get("content", ""))
        raise RuntimeError(
            f"candidate codes absent from biased top-{self.top_logprobs}; text={text!r}"
        )

    def _score_candidates(
        self, content: list[dict[str, str]], candidates: list[str]
    ) -> dict[str, float]:
        request_fingerprint = {
            "model": self.model,
            "detail": self.detail,
            "reasoning_effort": self.reasoning_effort,
            "top_logprobs": self.top_logprobs,
            "max_output_tokens": self.max_output_tokens,
            "score_mode": self.score_mode,
            "content": content,
            "candidates": candidates,
        }
        key = hashlib.sha256(
            json.dumps(request_fingerprint, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return {candidate: float(cached["scores"][candidate]) for candidate in candidates}
        with self._lock:
            if self._call_count >= self.max_api_calls:
                raise RuntimeError(f"OpenAI API call limit reached: {self.max_api_calls}")
            self._call_count += 1
        if self.score_mode == "discrete":
            response = self.client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": content}],
                reasoning={"effort": self.reasoning_effort},
                max_output_tokens=self.max_output_tokens,
                store=False,
                text={"verbosity": "low"},
            )
            payload = response.model_dump(mode="json")
            output_text = str(getattr(response, "output_text", ""))
            selected = self._normalized_token(output_text)
            if selected not in candidates:
                raise RuntimeError(
                    f"OpenAI discrete response is not a candidate code: {output_text!r}"
                )
            scores = {candidate: (0.0 if candidate == selected else -1.0) for candidate in candidates}
            cache_path.write_text(
                json.dumps(
                    {
                        "scores": scores,
                        "score_mode": "discrete",
                        "response_id": payload.get("id"),
                        "model": payload.get("model"),
                        "usage": payload.get("usage"),
                        "output": payload.get("output"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            return scores
        if self.score_mode != "logprobs":
            raise ValueError(f"unsupported OpenAI score mode: {self.score_mode}")
        try:
            import tiktoken
        except ImportError as error:
            raise RuntimeError("OpenAI candidate scoring requires the `tiktoken` package") from error
        encoding = tiktoken.get_encoding("o200k_base")
        logit_bias: dict[str, int] = {}
        for candidate in candidates:
            for token_text in (candidate, f" {candidate}"):
                token_ids = encoding.encode(token_text)
                if len(token_ids) != 1:
                    raise RuntimeError(f"candidate code is not one token: {token_text!r}")
                logit_bias[str(token_ids[0])] = 100
        chat_content: list[dict[str, Any]] = []
        for item in content:
            if item["type"] == "input_text":
                chat_content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "input_image":
                chat_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": item["image_url"], "detail": item["detail"]},
                    }
                )
            else:
                raise ValueError(f"unsupported OpenAI content item: {item['type']}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": chat_content}],
            reasoning_effort=self.reasoning_effort,
            max_completion_tokens=self.max_output_tokens,
            logprobs=True,
            top_logprobs=self.top_logprobs,
            logit_bias=logit_bias,
            store=False,
            verbosity="low",
        )
        payload = response.model_dump(mode="json")
        scores = self._extract_chat_scores(payload, candidates)
        cache_path.write_text(
            json.dumps(
                {
                    "scores": scores,
                    "response_id": payload.get("id"),
                    "model": payload.get("model"),
                    "usage": payload.get("usage"),
                    "choices": payload.get("choices"),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return scores

    def _map_parallel(self, function: Any, items: list[Any]) -> list[Any]:
        if self.workers <= 1:
            return [function(item) for item in items]
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            return list(executor.map(function, items))

    def class_scores(
        self, image: str, labels: list[str], prompt_variant: int = 0
    ) -> dict[str, float]:
        prompt, codes = class_prompt(labels, prompt_variant)
        content = [self._image_item(image), {"type": "input_text", "text": prompt}]
        code_scores = self._score_candidates(content, [codes[label] for label in labels])
        return {label: code_scores[codes[label]] for label in labels}

    def class_scores_batch(
        self, images: list[str], labels: list[str], prompt_variant: int = 0
    ) -> list[dict[str, float]]:
        return self._map_parallel(
            lambda image: self.class_scores(image, labels, prompt_variant), images
        )

    def discover(
        self,
        images_a: list[str],
        images_b: list[str],
        label_a: str,
        label_b: str,
        source_mode: str,
        atoms_per_bootstrap: int,
    ) -> str:
        raise NotImplementedError("OpenAI pilot backend currently supports scoring, not atom discovery")

    def atom_probabilities(
        self, image: str, atom: DecisionAtom, prompt_variant: int
    ) -> np.ndarray:
        content = [
            self._image_item(image),
            {"type": "input_text", "text": atom_prompt(atom, prompt_variant)},
        ]
        scores = self._score_candidates(content, ["A", "B", "U"])
        return softmax(np.asarray([scores[code] for code in ("A", "B", "U")]))

    def icl_scores(
        self, query: Sample, demos: list[Sample], labels: list[str]
    ) -> dict[str, float]:
        demo_texts, query_text, codes = icl_prompt(labels, [demo.label for demo in demos])
        content: list[dict[str, str]] = []
        for demo, text in zip(demos, demo_texts):
            content.extend([self._image_item(demo.image), {"type": "input_text", "text": text}])
        content.extend([self._image_item(query.image), {"type": "input_text", "text": query_text}])
        code_scores = self._score_candidates(content, [codes[label] for label in labels])
        return {label: code_scores[codes[label]] for label in labels}

    def icl_scores_batch(
        self, queries: list[Sample], demos: list[list[Sample]], labels: list[str]
    ) -> list[dict[str, float]]:
        return self._map_parallel(
            lambda item: self.icl_scores(item[0], item[1], labels), list(zip(queries, demos))
        )

    def task_scores(self, query: Sample, demos: list[Sample], labels: list[str]) -> dict[str, float]:
        abstract, codes = task_icl_content(query, demos, labels)
        content: list[dict[str, str]] = []
        for block in abstract:
            if block["type"] == "image":
                content.append(self._image_item(str(block["image"])))
            else:
                content.append({"type": "input_text", "text": str(block["text"])})
        options = list(query.options) if query.options else list(labels)
        values = self._score_candidates(content, [codes[option] for option in options])
        return dict(zip(options, values))

class Qwen3VLBackend(TargetVLM):
    def __init__(self, config: dict[str, Any]):
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError("Qwen backend requires the optional 'hf' dependencies") from error
        self.torch = torch
        self.process_vision_info = process_vision_info
        model_name = config["model"]
        dtype_name = str(config.get("dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name) if dtype_name != "auto" else "auto"
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": config.get("device_map", "auto"),
            "local_files_only": bool(config.get("local_files_only", True)),
            "attn_implementation": config.get("attn_implementation", "sdpa"),
        }
        quantization = str(config.get("quantization", "none")).lower()
        if quantization in ("4bit", "nf4"):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype if dtype != "auto" else torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization != "none":
            raise ValueError(f"unsupported Qwen quantization mode: {quantization}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            local_files_only=bool(config.get("local_files_only", True)),
        )
        self.max_new_tokens = int(config.get("max_new_tokens", 768))
        self.atom_batch_size = int(config.get("atom_batch_size", 8))
        self.classification_batch_size = int(config.get("classification_batch_size", 4))
        self.icl_batch_size = int(config.get("icl_batch_size", 2))
        self.efficient_next_logits = bool(config.get("efficient_next_logits", True))
        self.share_atom_vision_features = bool(config.get("share_atom_vision_features", True))
        self.min_pixels = int(config.get("min_pixels", 65_536))
        self.max_pixels = int(config.get("max_pixels", 401_408))

    def _next_token_logits(self, inputs: dict[str, Any]) -> Any:
        """Return each batch row's final non-padding logits without materializing all positions."""
        device = inputs["input_ids"].device
        positions = (
            self.torch.arange(inputs["input_ids"].shape[1], device=device)[None, :]
            * inputs["attention_mask"].long()
        ).max(dim=1).values
        with self.torch.inference_mode():
            if self.efficient_next_logits:
                # Qwen accepts explicit sequence positions here. Passing one position per
                # batch row produces B x B x V logits instead of B x sequence_length x V.
                # The diagonal contains the requested position for each corresponding row.
                logits = self.model(
                    **inputs,
                    use_cache=False,
                    logits_to_keep=positions,
                ).logits.float()
                batch_indices = self.torch.arange(len(positions), device=device)
                return logits[batch_indices, batch_indices]
            logits = self.model(**inputs, use_cache=False).logits.float()
        return logits[self.torch.arange(len(positions), device=device), positions]

    def _image_block(self, image: str) -> dict[str, Any]:
        return {
            "type": "image",
            "image": image,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }

    def _prepare(self, content: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.process_vision_info(messages)
        inputs = self.processor(
            text=[rendered], images=images, videos=videos, padding=True, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    def _score_candidates(self, content: list[dict[str, Any]], candidates: list[str]) -> list[float]:
        inputs = self._prepare(content)
        prompt_length = int(inputs["input_ids"].shape[1])
        encoded = [
            self.processor.tokenizer(candidate, add_special_tokens=False, return_tensors="pt")["input_ids"].to(
                inputs["input_ids"].device
            )
            for candidate in candidates
        ]
        if all(token_ids.shape[1] == 1 for token_ids in encoded):
            logits = self._next_token_logits(inputs)[0]
            log_probs = logits.log_softmax(dim=-1)
            return [float(log_probs[token_ids[0, 0]].item()) for token_ids in encoded]
        results: list[float] = []
        with self.torch.inference_mode():
            for token_ids in encoded:
                full_inputs = dict(inputs)
                full_inputs["input_ids"] = self.torch.cat([inputs["input_ids"], token_ids], dim=1)
                full_inputs["attention_mask"] = self.torch.cat(
                    [inputs["attention_mask"], self.torch.ones_like(token_ids)], dim=1
                )
                logits = self.model(**full_inputs, use_cache=False).logits
                positions = logits[0, prompt_length - 1 : prompt_length + token_ids.shape[1] - 1]
                log_probs = positions.float().log_softmax(dim=-1)
                chosen = log_probs.gather(1, token_ids[0].unsqueeze(1)).squeeze(1)
                results.append(float(chosen.mean().item()))
        return results

    def _generate(self, content: list[dict[str, Any]]) -> str:
        inputs = self._prepare(content)
        prompt_length = int(inputs["input_ids"].shape[1])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return self.processor.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def _shared_image_next_logits(
        self, messages_batch: list[list[dict[str, Any]]]
    ) -> tuple[Any, Any]:
        """Run one vision encoding for many text questions about the same image."""
        rendered = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        image_inputs: list[Any] = []
        for messages in messages_batch:
            batch_images, _ = self.process_vision_info(messages)
            image_inputs.extend(batch_images or [])
        inputs = self.processor(text=rendered, images=image_inputs, padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

        batch_size = len(messages_batch)
        grid = inputs["image_grid_thw"]
        first_patch_count = int(grid[0].prod().item())
        with self.torch.inference_mode():
            vision = self.model.model.get_image_features(
                inputs["pixel_values"][:first_patch_count], grid[:1], return_dict=True
            )
            pooled = self.torch.cat(vision.pooler_output, dim=0)
            image_features = pooled.repeat(batch_size, 1)
            deepstack = [features.repeat(batch_size, 1) for features in vision.deepstack_features]

            input_ids = inputs["input_ids"]
            input_embeddings = self.model.model.get_input_embeddings()(input_ids)
            image_features = image_features.to(input_embeddings.device, input_embeddings.dtype)
            image_mask, _ = self.model.model.get_placeholder_mask(
                input_ids, inputs_embeds=input_embeddings, image_features=image_features
            )
            input_embeddings = input_embeddings.masked_scatter(image_mask, image_features)
            visual_mask = image_mask[..., 0]
            position_ids = self.model.model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=grid,
                inputs_embeds=input_embeddings,
                attention_mask=inputs["attention_mask"],
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
            )
            language_output = self.model.model.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=inputs["attention_mask"],
                inputs_embeds=input_embeddings,
                visual_pos_masks=visual_mask,
                deepstack_visual_embeds=deepstack,
                use_cache=False,
            )
            positions = (
                self.torch.arange(input_ids.shape[1], device=device)[None, :]
                * inputs["attention_mask"].long()
            ).max(dim=1).values
            hidden = language_output[0][self.torch.arange(batch_size, device=device), positions]
            logits = self.model.lm_head(hidden).float()
        return logits, inputs

    def class_scores(self, image: str, labels: list[str], prompt_variant: int = 0) -> dict[str, float]:
        prompt, codes = class_prompt(labels, prompt_variant)
        content = [self._image_block(image), {"type": "text", "text": prompt}]
        values = self._score_candidates(content, [codes[label] for label in labels])
        return dict(zip(labels, values))

    def class_scores_batch(
        self, images: list[str], labels: list[str], prompt_variant: int = 0
    ) -> list[dict[str, float]]:
        prompt, codes = class_prompt(labels, prompt_variant)
        code_rows = [
            self.processor.tokenizer(codes[label], add_special_tokens=False)["input_ids"] for label in labels
        ]
        if not all(len(row) == 1 for row in code_rows):
            return super().class_scores_batch(images, labels, prompt_variant)
        device = next(self.model.parameters()).device
        code_ids = self.torch.tensor([row[0] for row in code_rows], device=device)
        output: list[dict[str, float]] = []
        for start in range(0, len(images), self.classification_batch_size):
            image_batch = images[start : start + self.classification_batch_size]
            messages_batch = [
                [{"role": "user", "content": [self._image_block(image), {"type": "text", "text": prompt}]}]
                for image in image_batch
            ]
            rendered = [
                self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in messages_batch
            ]
            image_inputs: list[Any] = []
            for messages in messages_batch:
                batch_images, _ = self.process_vision_info(messages)
                image_inputs.extend(batch_images or [])
            inputs = self.processor(text=rendered, images=image_inputs, padding=True, return_tensors="pt")
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            next_logits = self._next_token_logits(inputs)
            candidate_scores = next_logits[:, code_ids].log_softmax(dim=1).cpu().numpy()
            output.extend(
                {label: float(row[index]) for index, label in enumerate(labels)}
                for row in candidate_scores
            )
        return output

    def discover(
        self,
        images_a: list[str],
        images_b: list[str],
        label_a: str,
        label_b: str,
        source_mode: str,
        atoms_per_bootstrap: int,
    ) -> str:
        content: list[dict[str, Any]] = []
        for index, image in enumerate(images_a, start=1):
            content.extend(
                [{"type": "text", "text": f"Group A image {index}:"}, self._image_block(image)]
            )
        for index, image in enumerate(images_b, start=1):
            content.extend(
                [{"type": "text", "text": f"Group B image {index}:"}, self._image_block(image)]
            )
        content.append(
            {
                "type": "text",
                "text": discovery_prompt(label_a, label_b, atoms_per_bootstrap, source_mode),
            }
        )
        return self._generate(content)

    def atom_probabilities(self, image: str, atom: DecisionAtom, prompt_variant: int) -> np.ndarray:
        content = [
            self._image_block(image),
            {"type": "text", "text": atom_prompt(atom, prompt_variant)},
        ]
        return softmax(np.asarray(self._score_candidates(content, ["A", "B", "U"])))

    def atom_probabilities_batch(
        self, image: str, atoms: list[DecisionAtom], prompt_variant: int
    ) -> np.ndarray:
        # Qwen's processor can batch repeated images with distinct text prompts.
        # This preserves one A/B/U logit readout per atom while amortizing GPU work.
        if self.atom_batch_size <= 1:
            return super().atom_probabilities_batch(image, atoms, prompt_variant)
        output: list[np.ndarray] = []
        token_rows = [
            self.processor.tokenizer(code, add_special_tokens=False)["input_ids"] for code in ("A", "B", "U")
        ]
        if not all(len(row) == 1 for row in token_rows):
            return super().atom_probabilities_batch(image, atoms, prompt_variant)
        token_ids = self.torch.tensor(
            [row[0] for row in token_rows], device=next(self.model.parameters()).device
        )
        for start in range(0, len(atoms), self.atom_batch_size):
            batch = atoms[start : start + self.atom_batch_size]
            messages_batch = [
                [
                    {
                        "role": "user",
                        "content": [
                            self._image_block(image),
                            {"type": "text", "text": atom_prompt(atom, prompt_variant)},
                        ],
                    }
                ]
                for atom in batch
            ]
            if self.share_atom_vision_features:
                next_logits, _ = self._shared_image_next_logits(messages_batch)
            else:
                rendered = [
                    self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    for messages in messages_batch
                ]
                image_inputs: list[Any] = []
                for messages in messages_batch:
                    images, _ = self.process_vision_info(messages)
                    image_inputs.extend(images or [])
                inputs = self.processor(text=rendered, images=image_inputs, padding=True, return_tensors="pt")
                device = next(self.model.parameters()).device
                inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
                next_logits = self._next_token_logits(inputs)
            selected = next_logits[:, token_ids].softmax(dim=1).cpu().numpy()
            output.extend(selected)
        return np.asarray(output, dtype=np.float64)

    def icl_scores(self, query: Sample, demos: list[Sample], labels: list[str]) -> dict[str, float]:
        demo_texts, query_text, codes = icl_prompt(labels, [demo.label for demo in demos])
        content: list[dict[str, Any]] = []
        for demo, text in zip(demos, demo_texts):
            content.extend([self._image_block(demo.image), {"type": "text", "text": text}])
        content.extend([self._image_block(query.image), {"type": "text", "text": query_text}])
        values = self._score_candidates(content, [codes[label] for label in labels])
        return dict(zip(labels, values))

    def icl_scores_batch(
        self, queries: list[Sample], demos: list[list[Sample]], labels: list[str]
    ) -> list[dict[str, float]]:
        if len(queries) != len(demos):
            raise ValueError("queries and demo lists must align")
        _, _, codes = icl_prompt(labels, [])
        code_rows = [
            self.processor.tokenizer(codes[label], add_special_tokens=False)["input_ids"] for label in labels
        ]
        if not all(len(row) == 1 for row in code_rows):
            return super().icl_scores_batch(queries, demos, labels)
        device = next(self.model.parameters()).device
        code_ids = self.torch.tensor([row[0] for row in code_rows], device=device)
        output: list[dict[str, float]] = []
        for start in range(0, len(queries), self.icl_batch_size):
            query_batch = queries[start : start + self.icl_batch_size]
            demo_batch = demos[start : start + self.icl_batch_size]
            messages_batch: list[list[dict[str, Any]]] = []
            for query, query_demos in zip(query_batch, demo_batch):
                demo_texts, query_text, _ = icl_prompt(labels, [demo.label for demo in query_demos])
                content: list[dict[str, Any]] = []
                for demo, text in zip(query_demos, demo_texts):
                    content.extend([self._image_block(demo.image), {"type": "text", "text": text}])
                content.extend([self._image_block(query.image), {"type": "text", "text": query_text}])
                messages_batch.append([{"role": "user", "content": content}])
            rendered = [
                self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in messages_batch
            ]
            image_inputs: list[Any] = []
            for messages in messages_batch:
                batch_images, _ = self.process_vision_info(messages)
                image_inputs.extend(batch_images or [])
            inputs = self.processor(text=rendered, images=image_inputs, padding=True, return_tensors="pt")
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            next_logits = self._next_token_logits(inputs)
            candidate_scores = next_logits[:, code_ids].log_softmax(dim=1).cpu().numpy()
            output.extend(
                {label: float(row[index]) for index, label in enumerate(labels)}
                for row in candidate_scores
            )
        return output

    def task_scores(self, query: Sample, demos: list[Sample], labels: list[str]) -> dict[str, float]:
        abstract, codes = task_icl_content(query, demos, labels)
        content = [
            self._image_block(str(block["image"]))
            if block["type"] == "image"
            else {"type": "text", "text": str(block["text"])}
            for block in abstract
        ]
        options = list(query.options) if query.options else list(labels)
        values = self._score_candidates(content, [codes[option] for option in options])
        return dict(zip(options, values))

    def generate_task_answer(self, query: Sample, demos: list[Sample], labels: list[str]) -> str:
        abstract, _ = task_icl_content(query, demos, labels)
        content = [
            self._image_block(str(block["image"]))
            if block["type"] == "image"
            else {"type": "text", "text": str(block["text"])}
            for block in abstract
        ]
        return self._generate(content).strip()


class HFImageTextBackend(TargetVLM):
    """Transformers-native frozen VLM backend shared by MiMo-VL and GLM-V."""

    def __init__(self, config: dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("generic HF image-text backend dependencies are missing") from error
        self.torch = torch
        self.model_name = str(config["model"])
        dtype_name = str(config.get("dtype", "bfloat16"))
        self.dtype = getattr(torch, dtype_name)
        local_only = bool(config.get("local_files_only", True))
        trust_remote_code = bool(config.get("trust_remote_code", False))
        load_kwargs: dict[str, Any] = {
            "device_map": config.get("device_map", "auto"),
            "dtype": self.dtype,
            "local_files_only": local_only,
            "trust_remote_code": trust_remote_code,
        }
        quantization = str(config.get("quantization", "none")).lower()
        if quantization in {"4bit", "nf4"}:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization in {"8bit", "int8"}:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization not in {"none", "bf16", "fp16"}:
            raise ValueError(f"unsupported HF image-text quantization: {quantization}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, **load_kwargs
        ).eval()

        processor_kwargs: dict[str, Any] = {
            "local_files_only": local_only,
            "trust_remote_code": trust_remote_code,
        }
        shortest = config.get("image_size_shortest_edge")
        longest = config.get("image_size_longest_edge")
        if shortest is not None or longest is not None:
            if shortest is None or longest is None:
                raise ValueError("both image size edges must be configured")
            processor_kwargs["size"] = {
                "shortest_edge": int(shortest),
                "longest_edge": int(longest),
            }
        min_pixels = config.get("processor_min_pixels", config.get("min_pixels"))
        max_pixels = config.get("processor_max_pixels", config.get("max_pixels"))
        if min_pixels is not None or max_pixels is not None:
            if min_pixels is None or max_pixels is None:
                raise ValueError("both processor pixel bounds must be configured")
            processor_kwargs["min_pixels"] = int(min_pixels)
            processor_kwargs["max_pixels"] = int(max_pixels)
        self.processor = AutoProcessor.from_pretrained(self.model_name, **processor_kwargs)
        if getattr(self.processor, "tokenizer", None) is None:
            raise ValueError(f"processor for {self.model_name} has no tokenizer")
        self.processor.tokenizer.padding_side = "left"
        self.enable_thinking = config.get("enable_thinking")
        self.prompt_suffix = str(config.get("prompt_suffix", "")).strip()
        self.drop_token_type_ids = bool(config.get("drop_token_type_ids", False))
        self.candidate_prefix = str(config.get("candidate_prefix", ""))
        self.downsample_mode = config.get("downsample_mode")
        self.max_slice_nums = config.get("max_slice_nums")
        self.max_new_tokens = int(config.get("max_new_tokens", 16))

    @property
    def device(self) -> Any:
        return next(
            parameter for parameter in self.model.parameters() if parameter.device.type != "meta"
        ).device

    @staticmethod
    def _image_content(image: str) -> dict[str, Any]:
        return {"type": "image", "url": str(Path(image).resolve())}

    def _messages(self, query: Sample, demos: list[Sample], labels: list[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
        abstract, codes = task_icl_content(query, demos, labels)
        content = [
            self._image_content(str(block["image"]))
            if block["type"] == "image"
            else {"type": "text", "text": str(block["text"])}
            for block in abstract
        ]
        if self.prompt_suffix:
            content[-1]["text"] = str(content[-1]["text"]) + f" {self.prompt_suffix}"
        return [{"role": "user", "content": content}], codes

    def _prepare(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = bool(self.enable_thinking)
        processor_kwargs: dict[str, Any] = {}
        if self.downsample_mode is not None:
            processor_kwargs["downsample_mode"] = self.downsample_mode
        if self.max_slice_nums is not None:
            processor_kwargs["max_slice_nums"] = int(self.max_slice_nums)
        if processor_kwargs:
            kwargs["processor_kwargs"] = processor_kwargs
        inputs = self.processor.apply_chat_template(messages, **kwargs)
        prepared = dict(inputs.to(self.device))
        if self.drop_token_type_ids:
            prepared.pop("token_type_ids", None)
        return prepared

    def _candidate_logits(self, inputs: dict[str, Any], candidates: list[str]) -> list[float]:
        prefix_ids: list[int] = []
        if self.candidate_prefix:
            prefix_ids = list(
                self.processor.tokenizer(
                    self.candidate_prefix, add_special_tokens=False
                )["input_ids"]
            )
        token_ids: list[int] = []
        for candidate in candidates:
            text = self.candidate_prefix + candidate if self.candidate_prefix else candidate
            encoded = list(
                self.processor.tokenizer(text, add_special_tokens=False)["input_ids"]
            )
            if prefix_ids:
                if encoded[: len(prefix_ids)] != prefix_ids:
                    raise ValueError("candidate prefix changes tokenizer boundary")
                encoded = encoded[len(prefix_ids) :]
            if len(encoded) != 1:
                raise ValueError(
                    f"choice code {candidate!r} is not one token for {self.model_name}"
                )
            token_ids.append(int(encoded[0]))
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("choice codes map to duplicate tokenizer tokens")
        forward = dict(inputs)
        if prefix_ids:
            prefix = self.torch.tensor(
                [prefix_ids], dtype=forward["input_ids"].dtype, device=forward["input_ids"].device
            )
            forward["input_ids"] = self.torch.cat([forward["input_ids"], prefix], dim=1)
            if "attention_mask" in forward:
                attention = self.torch.ones(
                    (forward["attention_mask"].shape[0], len(prefix_ids)),
                    dtype=forward["attention_mask"].dtype,
                    device=forward["attention_mask"].device,
                )
                forward["attention_mask"] = self.torch.cat(
                    [forward["attention_mask"], attention], dim=1
                )
            if "token_type_ids" in forward:
                token_types = forward["token_type_ids"][:, -1:].expand(-1, len(prefix_ids))
                forward["token_type_ids"] = self.torch.cat(
                    [forward["token_type_ids"], token_types], dim=1
                )
            if "mm_token_type_ids" in forward:
                text_types = self.torch.zeros(
                    (forward["mm_token_type_ids"].shape[0], len(prefix_ids)),
                    dtype=forward["mm_token_type_ids"].dtype,
                    device=forward["mm_token_type_ids"].device,
                )
                forward["mm_token_type_ids"] = self.torch.cat(
                    [forward["mm_token_type_ids"], text_types], dim=1
                )
        forward["use_cache"] = False
        if self.downsample_mode is not None:
            forward["downsample_mode"] = self.downsample_mode
        with self.torch.inference_mode():
            logits = self.model(**forward).logits[0, -1].float().log_softmax(dim=-1)
        return [float(logits[token_id].item()) for token_id in token_ids]

    def task_scores(self, query: Sample, demos: list[Sample], labels: list[str]) -> dict[str, float]:
        messages, codes = self._messages(query, demos, labels)
        options = list(query.options) if query.options else list(labels)
        values = self._candidate_logits(
            self._prepare(messages), [codes[option] for option in options]
        )
        return dict(zip(options, values))

    def generate_task_answer(self, query: Sample, demos: list[Sample], labels: list[str]) -> str:
        messages, _ = self._messages(query, demos, labels)
        inputs = self._prepare(messages)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True
            )
        source_length = int(inputs["input_ids"].shape[1])
        return self.processor.tokenizer.decode(
            generated[0, source_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def class_scores(self, image: str, labels: list[str], prompt_variant: int = 0) -> dict[str, float]:
        query = Sample(id="query", split="test", image=image, label=labels[0])
        return self.task_scores(query, [], labels)

    def icl_scores(self, query: Sample, demos: list[Sample], labels: list[str]) -> dict[str, float]:
        return self.task_scores(query, demos, labels)

    def discover(
        self,
        images_a: list[str],
        images_b: list[str],
        label_a: str,
        label_b: str,
        source_mode: str,
        atoms_per_bootstrap: int,
    ) -> str:
        raise NotImplementedError("generic paper backend does not perform atom discovery")

    def atom_probabilities(self, image: str, atom: DecisionAtom, prompt_variant: int) -> np.ndarray:
        raise NotImplementedError("generic paper backend does not measure decision atoms")


class Siglip2Retriever(VisualRetriever):
    def __init__(self, config: dict[str, Any]):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError("SigLIP2 retriever requires torch and transformers") from error
        self.torch = torch
        model_name = config["model"]
        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=bool(config.get("local_files_only", True))
        )
        dtype_name = str(config.get("dtype", "float16"))
        dtype = getattr(torch, dtype_name) if dtype_name != "auto" else None
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=bool(config.get("local_files_only", True)),
            dtype=dtype,
        ).to(config.get("device", "cuda")).eval()
        self.device = config.get("device", "cuda")
        self.batch_size = int(config.get("batch_size", 32))

    def embed(self, images: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = [_open_rgb_image(path) for path in images[start : start + self.batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                features = self.model.get_image_features(**inputs)
            # transformers 5 returns a ModelOutput here for SigLIP2, while
            # earlier releases returned the pooled tensor directly.
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            rows.append(features.float().cpu().numpy())
        return np.concatenate(rows, axis=0)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self.processor(
                text=batch, padding="max_length", truncation=True, return_tensors="pt"
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                features = self.model.get_text_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            rows.append(features.float().cpu().numpy())
        return np.concatenate(rows, axis=0)


def build_target(config: dict[str, Any]) -> TargetVLM:
    kind = config.get("kind", "qwen3_vl")
    if kind == "mock":
        return MockTargetVLM()
    if kind == "qwen3_vl":
        return Qwen3VLBackend(config)
    if kind == "openai_responses":
        return OpenAIResponsesBackend(config)
    if kind == "hf_image_text":
        return HFImageTextBackend(config)
    raise ValueError(f"unsupported target backend: {kind}")


def build_retriever(config: dict[str, Any]) -> VisualRetriever:
    kind = config.get("kind", "siglip2")
    if kind == "mock":
        return MockRetriever()
    if kind == "siglip2":
        return Siglip2Retriever(config)
    raise ValueError(f"unsupported retriever backend: {kind}")

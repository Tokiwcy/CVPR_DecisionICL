from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import os
from pathlib import Path
import sys
from typing import Protocol, Sequence

import numpy as np


class JointEncoder(Protocol):
    model_id: str
    model_revision: str

    def encode_images(self, image_paths: Sequence[str]) -> np.ndarray: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...

    def logit_scale_and_bias(self) -> tuple[float, float]: ...


@dataclass
class SiglipEncoder:
    model_id: str
    model_revision: str = "local"
    device: str = "cuda"
    batch_size: int = 32
    dtype: str = "float16"

    def __post_init__(self) -> None:
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.dtype]
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=None if self.model_revision == "local" else self.model_revision,
            local_files_only=Path(self.model_id).exists(),
        )
        self._model = AutoModel.from_pretrained(
            self.model_id,
            revision=None if self.model_revision == "local" else self.model_revision,
            dtype=dtype,
            local_files_only=Path(self.model_id).exists(),
        ).eval().to(self.device)

    @staticmethod
    def _normalize(features: "object") -> np.ndarray:
        import torch

        if isinstance(features, torch.Tensor):
            tensor = features
        else:
            tensor = None
            for attribute in ("image_embeds", "text_embeds", "pooler_output"):
                candidate = getattr(features, attribute, None)
                if isinstance(candidate, torch.Tensor):
                    tensor = candidate
                    break
            if tensor is None:
                raise TypeError(f"unsupported encoder feature output: {type(features).__name__}")
        tensor = tensor.float()
        tensor = tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    def encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        import torch
        from PIL import Image

        self._load()
        assert self._processor is not None and self._model is not None
        batches: list[np.ndarray] = []
        for start in range(0, len(image_paths), self.batch_size):
            images = []
            for raw_path in image_paths[start : start + self.batch_size]:
                with Image.open(raw_path) as image:
                    images.append(image.convert("RGB"))
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = self._model.get_image_features(**inputs)
            batches.append(self._normalize(features))
        if not batches:
            raise ValueError("cannot encode an empty image list")
        return np.concatenate(batches, axis=0)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        self._load()
        assert self._processor is not None and self._model is not None
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            inputs = self._processor(
                text=list(texts[start : start + self.batch_size]),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = self._model.get_text_features(**inputs)
            batches.append(self._normalize(features))
        if not batches:
            raise ValueError("cannot encode an empty text list")
        return np.concatenate(batches, axis=0)

    def logit_scale_and_bias(self) -> tuple[float, float]:
        import torch

        self._load()
        assert self._model is not None
        raw_scale = getattr(self._model, "logit_scale", None)
        raw_bias = getattr(self._model, "logit_bias", None)
        if raw_scale is None:
            raise RuntimeError("the loaded model does not expose SigLIP logit_scale")
        with torch.no_grad():
            scale = float(raw_scale.detach().float().exp().cpu().item())
            bias = 0.0 if raw_bias is None else float(raw_bias.detach().float().cpu().item())
        return scale, bias


@dataclass
class FlairEncoder:
    """Adapter for the official ExplainableML/FLAIR text-conditioned scorer."""

    model_id: str
    source_dir: str
    model_revision: str
    checkpoint_sha256: str
    device: str = "cuda"
    batch_size: int = 8
    dtype: str = "float16"
    architecture: str = "ViT-B-16-FLAIR"

    def __post_init__(self) -> None:
        self.model_revision = (
            f"{self.model_revision}|checkpoint-sha256={self.checkpoint_sha256.lower()}"
        )
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    @property
    def uncalibrated_evidence_transform(self) -> str:
        return (
            "flair-text-conditioned-native-scale__uncalibrated__"
            "group-softmax__one-vs-rest-log-margin-v1"
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        checkpoint = Path(self.model_id).resolve()
        source = Path(self.source_dir).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FLAIR checkpoint not found: {checkpoint}")
        if not (source / "flair" / "__init__.py").is_file():
            raise FileNotFoundError(f"official FLAIR source package not found: {source}")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != self.checkpoint_sha256.lower():
            raise ValueError(
                f"FLAIR checkpoint SHA-256 mismatch: expected {self.checkpoint_sha256}, "
                f"got {actual_sha256}"
            )
        source_text = str(source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        flair = importlib.import_module("flair")
        module_path = Path(str(flair.__file__)).resolve()
        if source not in module_path.parents:
            raise RuntimeError(
                f"imported an unexpected flair package from {module_path}; expected under {source}"
            )
        # The official CVPR 2025 checkpoint contains legacy NumPy scalar
        # metadata. PyTorch >=2.6 requires explicitly opting out of
        # weights_only loading. We do so only after the exact checkpoint hash
        # has been verified above.
        previous_force = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            model, _, preprocess = flair.create_model_and_transforms(
                self.architecture,
                pretrained=str(checkpoint),
                precision="fp32",
                device=self.device,
            )
        finally:
            if previous_force is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_force
        self._model = model.eval()
        self._preprocess = preprocess
        self._tokenizer = flair.get_tokenizer(self.architecture)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("FLAIR requested CUDA but CUDA is unavailable")

    def _autocast(self):
        import torch

        if not self.device.startswith("cuda") or self.dtype == "float32":
            return torch.autocast(device_type="cpu", enabled=False)
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(self.dtype)
        if dtype is None:
            raise ValueError(f"unsupported FLAIR dtype: {self.dtype}")
        return torch.autocast(device_type="cuda", dtype=dtype)

    @staticmethod
    def _normalized_numpy(tensor: "object") -> np.ndarray:
        import torch

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"expected a tensor, got {type(tensor).__name__}")
        values = tensor.float()
        values = values / values.norm(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).eps
        )
        return values.detach().cpu().numpy().astype(np.float32, copy=False)

    def _image_batch(self, image_paths: Sequence[str]):
        import torch
        from PIL import Image, ImageOps

        assert self._preprocess is not None
        rows = []
        for raw_path in image_paths:
            with Image.open(raw_path) as image:
                rows.append(self._preprocess(ImageOps.exif_transpose(image).convert("RGB")))
        return torch.stack(rows).to(self.device)

    def encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        import torch

        self._load()
        assert self._model is not None
        batches: list[np.ndarray] = []
        for start in range(0, len(image_paths), self.batch_size):
            image = self._image_batch(image_paths[start : start + self.batch_size])
            with torch.inference_mode(), self._autocast():
                global_image, _ = self._model.encode_image(image)
                global_image = self._model.image_post(global_image)
            batches.append(self._normalized_numpy(global_image))
        if not batches:
            raise ValueError("cannot encode an empty image list")
        return np.concatenate(batches, axis=0)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        self._load()
        assert self._model is not None and self._tokenizer is not None
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            tokens = self._tokenizer(list(texts[start : start + self.batch_size])).to(self.device)
            with torch.inference_mode(), self._autocast():
                global_text, _ = self._model.encode_text(tokens)
                global_text = self._model.text_post(global_text)
            batches.append(self._normalized_numpy(global_text))
        if not batches:
            raise ValueError("cannot encode an empty text list")
        return np.concatenate(batches, axis=0)

    def score_image_text_logits(
        self,
        image_paths: Sequence[str],
        texts: Sequence[str],
    ) -> np.ndarray:
        import torch

        self._load()
        assert self._model is not None and self._tokenizer is not None
        if not texts:
            raise ValueError("cannot score an empty text list")
        tokens = self._tokenizer(list(texts)).to(self.device)
        batches: list[np.ndarray] = []
        for start in range(0, len(image_paths), self.batch_size):
            image = self._image_batch(image_paths[start : start + self.batch_size])
            with torch.inference_mode(), self._autocast():
                output = self._model.get_logits(image=image, text=tokens)
                logits = output[0] if isinstance(output, tuple) else output
            batches.append(logits.float().detach().cpu().numpy().astype(np.float32, copy=False))
        if not batches:
            raise ValueError("cannot score an empty image list")
        return np.concatenate(batches, axis=0)

    def logit_scale_and_bias(self) -> tuple[float, float]:
        import torch

        self._load()
        assert self._model is not None
        with torch.no_grad():
            scale = float(self._model.logit_scale.detach().float().exp().cpu().item())
            bias = float(self._model.logit_bias.detach().float().cpu().item())
        return scale, bias

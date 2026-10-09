from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.special import logsumexp, softmax

from .hashing import array_sha256, stable_hash
from .io import read_json, write_json
from .schema import AtomBank, Sample


EVIDENCE_TRANSFORM = "siglip-native-scale__train-state-mean-debias__group-softmax__one-vs-rest-log-margin-v2"
UNCALIBRATED_EVIDENCE_TRANSFORM = "siglip-native-scale__uncalibrated__group-softmax__one-vs-rest-log-margin-v1"
DEFAULT_TEXT_TEMPLATE = "This image shows {description}."


@dataclass(frozen=True)
class EmbeddingStore:
    sample_ids: tuple[str, ...]
    embeddings: np.ndarray
    model_id: str
    model_revision: str

    def validate(self) -> None:
        values = np.asarray(self.embeddings)
        if values.ndim != 2 or values.shape[0] != len(self.sample_ids):
            raise ValueError("embedding rows must align with sample_ids")
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("embedding sample_ids must be unique")
        if not np.all(np.isfinite(values)):
            raise ValueError("embeddings contain non-finite values")
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-4):
            raise ValueError("embeddings must be L2-normalized")

    @property
    def store_hash(self) -> str:
        return stable_hash(
            {
                "sample_ids": list(self.sample_ids),
                "shape": list(self.embeddings.shape),
                "dtype": str(self.embeddings.dtype),
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "content_digest": array_sha256(np.asarray(self.embeddings, dtype=np.float32)),
            }
        )

    def save(self, prefix: str | Path) -> None:
        self.validate()
        prefix_path = Path(prefix)
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(str(prefix_path) + ".npz"), embeddings=np.asarray(self.embeddings, dtype=np.float32))
        write_json(
            Path(str(prefix_path) + ".json"),
            {
                "sample_ids": list(self.sample_ids),
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "store_hash": self.store_hash,
            },
        )

    @classmethod
    def load(cls, prefix: str | Path) -> "EmbeddingStore":
        prefix_path = Path(prefix)
        metadata = read_json(Path(str(prefix_path) + ".json"))
        with np.load(Path(str(prefix_path) + ".npz"), allow_pickle=False) as arrays:
            embeddings = arrays["embeddings"]
        store = cls(
            sample_ids=tuple(metadata["sample_ids"]),
            embeddings=embeddings,
            model_id=metadata["model_id"],
            model_revision=metadata["model_revision"],
        )
        store.validate()
        if metadata.get("store_hash") != store.store_hash:
            raise ValueError("embedding store hash mismatch")
        return store


@dataclass(frozen=True)
class EvidenceCalibration:
    state_ids: tuple[str, ...]
    state_logit_bias: tuple[float, ...]
    bank_hash: str
    model_id: str
    model_revision: str
    text_template: str
    source_split: str = "train"
    transform: str = EVIDENCE_TRANSFORM

    def validate(self) -> None:
        if self.source_split != "train":
            raise ValueError("evidence calibration must be fitted on train only")
        if len(self.state_ids) != len(self.state_logit_bias) or not self.state_ids:
            raise ValueError("evidence calibration state dimensions do not align")
        if len(self.state_ids) != len(set(self.state_ids)):
            raise ValueError("evidence calibration state_ids must be unique")
        if not np.all(np.isfinite(np.asarray(self.state_logit_bias, dtype=np.float64))):
            raise ValueError("evidence calibration contains non-finite biases")

    @property
    def calibration_hash(self) -> str:
        return stable_hash(
            {
                "state_ids": list(self.state_ids),
                "state_logit_bias": list(self.state_logit_bias),
                "bank_hash": self.bank_hash,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "text_template": self.text_template,
                "source_split": self.source_split,
                "transform": self.transform,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "state_ids": list(self.state_ids),
            "state_logit_bias": list(self.state_logit_bias),
            "bank_hash": self.bank_hash,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "text_template": self.text_template,
            "source_split": self.source_split,
            "transform": self.transform,
            "calibration_hash": self.calibration_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceCalibration":
        calibration = cls(
            state_ids=tuple(value["state_ids"]),
            state_logit_bias=tuple(float(x) for x in value["state_logit_bias"]),
            bank_hash=str(value["bank_hash"]),
            model_id=str(value["model_id"]),
            model_revision=str(value["model_revision"]),
            text_template=str(value["text_template"]),
            source_split=str(value.get("source_split", "train")),
            transform=str(value.get("transform", EVIDENCE_TRANSFORM)),
        )
        calibration.validate()
        if value.get("calibration_hash") != calibration.calibration_hash:
            raise ValueError("evidence calibration hash mismatch")
        return calibration

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceCalibration":
        return cls.from_dict(read_json(path))


@dataclass(frozen=True)
class EvidenceStore:
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    state_ids: tuple[str, ...]
    state_group_index: np.ndarray
    raw_logits: np.ndarray
    state_probabilities: np.ndarray
    signed_margins: np.ndarray
    ordinal_scores: np.ndarray
    applicability_scores: np.ndarray
    bank_hash: str
    model_id: str
    model_revision: str
    calibration_hash: str = "identity"
    text_template: str = DEFAULT_TEXT_TEMPLATE
    transform: str = UNCALIBRATED_EVIDENCE_TRANSFORM

    def validate(self) -> None:
        sample_count = len(self.sample_ids)
        state_count = len(self.state_ids)
        group_count = len(self.group_ids)
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("evidence sample_ids must be unique")
        if len(self.group_ids) != len(set(self.group_ids)):
            raise ValueError("evidence group_ids must be unique")
        if self.state_group_index.shape != (state_count,):
            raise ValueError("state_group_index shape mismatch")
        if np.any(self.state_group_index < 0) or np.any(self.state_group_index >= group_count):
            raise ValueError("state_group_index contains invalid group indices")
        expected_state_shape = (sample_count, state_count)
        for name, values in (
            ("raw_logits", self.raw_logits),
            ("state_probabilities", self.state_probabilities),
            ("signed_margins", self.signed_margins),
        ):
            if np.asarray(values).shape != expected_state_shape:
                raise ValueError(f"{name} shape mismatch")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite values")
        expected_group_shape = (sample_count, group_count)
        if self.ordinal_scores.shape != expected_group_shape:
            raise ValueError("ordinal_scores shape mismatch")
        if self.applicability_scores.shape != expected_group_shape:
            raise ValueError("applicability_scores shape mismatch")
        for group_index in range(group_count):
            mask = self.state_group_index == group_index
            sums = self.state_probabilities[:, mask].sum(axis=1)
            if not np.allclose(sums, 1.0, atol=1e-5):
                raise ValueError(f"group {self.group_ids[group_index]} probabilities do not sum to one")
        if np.any(self.applicability_scores < 0) or np.any(self.applicability_scores > 1):
            raise ValueError("applicability_scores must be in [0, 1]")

    def state_slice(self, group_index: int) -> np.ndarray:
        return np.flatnonzero(self.state_group_index == group_index)

    def group_probability(self, group_index: int) -> np.ndarray:
        return self.state_probabilities[:, self.state_slice(group_index)]

    def group_signed_margin(self, group_index: int) -> np.ndarray:
        return self.signed_margins[:, self.state_slice(group_index)]

    @property
    def store_hash(self) -> str:
        return stable_hash(
            {
                "sample_ids": list(self.sample_ids),
                "group_ids": list(self.group_ids),
                "state_ids": list(self.state_ids),
                "state_group_index": self.state_group_index.tolist(),
                "bank_hash": self.bank_hash,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "calibration_hash": self.calibration_hash,
                "text_template": self.text_template,
                "transform": self.transform,
                "content_digests": {
                    "raw_logits": array_sha256(self.raw_logits.astype(np.float32, copy=False)),
                    "state_probabilities": array_sha256(
                        self.state_probabilities.astype(np.float32, copy=False)
                    ),
                    "signed_margins": array_sha256(self.signed_margins.astype(np.float32, copy=False)),
                    "ordinal_scores": array_sha256(
                        np.nan_to_num(self.ordinal_scores, nan=9.876543).astype(np.float32, copy=False)
                    ),
                    "applicability_scores": array_sha256(
                        self.applicability_scores.astype(np.float32, copy=False)
                    ),
                },
            }
        )

    def save(self, prefix: str | Path) -> None:
        self.validate()
        prefix_path = Path(prefix)
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(str(prefix_path) + ".npz"),
            state_group_index=self.state_group_index.astype(np.int32),
            raw_logits=self.raw_logits.astype(np.float32),
            state_probabilities=self.state_probabilities.astype(np.float32),
            signed_margins=self.signed_margins.astype(np.float32),
            ordinal_scores=self.ordinal_scores.astype(np.float32),
            applicability_scores=self.applicability_scores.astype(np.float32),
        )
        write_json(
            Path(str(prefix_path) + ".json"),
            {
                "sample_ids": list(self.sample_ids),
                "group_ids": list(self.group_ids),
                "state_ids": list(self.state_ids),
                "bank_hash": self.bank_hash,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "calibration_hash": self.calibration_hash,
                "text_template": self.text_template,
                "transform": self.transform,
                "store_hash": self.store_hash,
            },
        )

    @classmethod
    def load(cls, prefix: str | Path) -> "EvidenceStore":
        prefix_path = Path(prefix)
        metadata = read_json(Path(str(prefix_path) + ".json"))
        with np.load(Path(str(prefix_path) + ".npz"), allow_pickle=False) as arrays:
            store = cls(
                sample_ids=tuple(metadata["sample_ids"]),
                group_ids=tuple(metadata["group_ids"]),
                state_ids=tuple(metadata["state_ids"]),
                state_group_index=arrays["state_group_index"],
                raw_logits=arrays["raw_logits"],
                state_probabilities=arrays["state_probabilities"],
                signed_margins=arrays["signed_margins"],
                ordinal_scores=arrays["ordinal_scores"],
                applicability_scores=arrays["applicability_scores"],
                bank_hash=metadata["bank_hash"],
                model_id=metadata["model_id"],
                model_revision=metadata["model_revision"],
                calibration_hash=metadata.get("calibration_hash", "identity"),
                text_template=metadata["text_template"],
                transform=metadata["transform"],
            )
        store.validate()
        if metadata.get("store_hash") != store.store_hash:
            raise ValueError("evidence store hash mismatch")
        return store


def build_embedding_store(samples: Sequence[Sample], encoder: Any) -> EmbeddingStore:
    embeddings = encoder.encode_images([sample.image_path for sample in samples])
    store = EmbeddingStore(
        sample_ids=tuple(sample.sample_id for sample in samples),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        model_id=encoder.model_id,
        model_revision=encoder.model_revision,
    )
    store.validate()
    return store


def _signed_one_vs_rest(logits: np.ndarray) -> np.ndarray:
    row_count, state_count = logits.shape
    if state_count < 2:
        raise ValueError("one-vs-rest evidence requires at least two states")
    result = np.empty_like(logits, dtype=np.float64)
    for state_index in range(state_count):
        others = np.delete(logits, state_index, axis=1)
        other_log_mean_exp = logsumexp(others, axis=1) - np.log(state_count - 1)
        result[:, state_index] = logits[:, state_index] - other_log_mean_exp
    return result


def _derive_group_arrays(
    *,
    raw_logits: np.ndarray,
    bank: AtomBank,
    state_group_array: np.ndarray,
    state_logit_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calibrated_logits = np.asarray(raw_logits, dtype=np.float64) - state_logit_bias[None, :]
    probabilities = np.zeros_like(calibrated_logits, dtype=np.float64)
    signed_margins = np.zeros_like(calibrated_logits, dtype=np.float64)
    ordinal = np.full((raw_logits.shape[0], len(bank.groups)), np.nan, dtype=np.float64)
    applicability = np.ones((raw_logits.shape[0], len(bank.groups)), dtype=np.float64)
    for group_index, group in enumerate(bank.groups):
        indices = np.flatnonzero(state_group_array == group_index)
        group_logits = calibrated_logits[:, indices]
        group_probabilities = softmax(group_logits, axis=1)
        probabilities[:, indices] = group_probabilities
        signed_margins[:, indices] = _signed_one_vs_rest(group_logits)
        abstain_local = np.asarray([state.abstain for state in group.states], dtype=bool)
        if np.any(abstain_local):
            applicability[:, group_index] = 1.0 - group_probabilities[:, abstain_local].sum(axis=1)
        if group.is_ordinal:
            substantive = ~abstain_local
            coordinates = np.asarray(
                [float(state.coordinate) for state in group.states if not state.abstain],
                dtype=np.float64,
            )
            minimum, maximum = float(coordinates.min()), float(coordinates.max())
            normalized_coordinates = 2.0 * (coordinates - minimum) / (maximum - minimum) - 1.0
            conditional = group_probabilities[:, substantive]
            conditional = conditional / conditional.sum(axis=1, keepdims=True).clip(min=1e-12)
            ordinal[:, group_index] = conditional @ normalized_coordinates
    return probabilities, signed_margins, ordinal, applicability


def build_evidence_store(
    *,
    samples: Sequence[Sample],
    image_embeddings: EmbeddingStore,
    bank: AtomBank,
    encoder: Any,
    text_template: str = DEFAULT_TEXT_TEMPLATE,
    calibration: EvidenceCalibration | None = None,
) -> EvidenceStore:
    bank.validate()
    image_embeddings.validate()
    sample_ids = tuple(sample.sample_id for sample in samples)
    if sample_ids != image_embeddings.sample_ids:
        raise ValueError("samples and image embeddings are not identically ordered")
    descriptions: list[str] = []
    state_ids: list[str] = []
    state_group_index: list[int] = []
    for group_index, group in enumerate(bank.groups):
        for state in group.states:
            descriptions.append(text_template.format(description=state.visual_description))
            state_ids.append(f"{group.group_id}/{state.state_id}")
            state_group_index.append(group_index)
    interaction = getattr(encoder, "score_image_text_logits", None)
    if callable(interaction):
        raw_logits = np.asarray(
            interaction([sample.image_path for sample in samples], descriptions),
            dtype=np.float32,
        )
        if raw_logits.shape != (len(samples), len(descriptions)):
            raise ValueError("text-conditioned image/text logits have the wrong shape")
    else:
        text_embeddings = encoder.encode_texts(descriptions)
        if text_embeddings.shape[1] != image_embeddings.embeddings.shape[1]:
            raise ValueError("image/text embedding dimensions do not match")
        scale, bias = encoder.logit_scale_and_bias()
        raw_logits = scale * (image_embeddings.embeddings @ text_embeddings.T) + bias
    state_group_array = np.asarray(state_group_index, dtype=np.int32)
    uncalibrated_transform = getattr(
        encoder,
        "uncalibrated_evidence_transform",
        UNCALIBRATED_EVIDENCE_TRANSFORM,
    )
    if calibration is None:
        state_logit_bias = np.zeros(len(state_ids), dtype=np.float64)
        calibration_hash = "identity"
        transform = uncalibrated_transform
    else:
        calibration.validate()
        expected = {
            "state_ids": tuple(state_ids),
            "bank_hash": bank.bank_hash,
            "model_id": encoder.model_id,
            "model_revision": encoder.model_revision,
            "text_template": text_template,
        }
        for name, value in expected.items():
            if getattr(calibration, name) != value:
                raise ValueError(f"evidence calibration mismatch: {name}")
        state_logit_bias = np.asarray(calibration.state_logit_bias, dtype=np.float64)
        calibration_hash = calibration.calibration_hash
        transform = calibration.transform
    probabilities, signed_margins, ordinal, applicability = _derive_group_arrays(
        raw_logits=raw_logits,
        bank=bank,
        state_group_array=state_group_array,
        state_logit_bias=state_logit_bias,
    )
    store = EvidenceStore(
        sample_ids=sample_ids,
        group_ids=tuple(group.group_id for group in bank.groups),
        state_ids=tuple(state_ids),
        state_group_index=state_group_array,
        raw_logits=np.asarray(raw_logits, dtype=np.float32),
        state_probabilities=np.asarray(probabilities, dtype=np.float32),
        signed_margins=np.asarray(signed_margins, dtype=np.float32),
        ordinal_scores=np.asarray(ordinal, dtype=np.float32),
        applicability_scores=np.asarray(applicability, dtype=np.float32),
        bank_hash=bank.bank_hash,
        model_id=encoder.model_id,
        model_revision=encoder.model_revision,
        calibration_hash=calibration_hash,
        text_template=text_template,
        transform=transform,
    )
    store.validate()
    return store


def fit_train_evidence_calibration(
    *,
    samples: Sequence[Sample],
    bank: AtomBank,
    uncalibrated_store: EvidenceStore,
) -> EvidenceCalibration:
    if any(sample.split != "train" for sample in samples):
        raise ValueError("evidence calibration may only read train samples")
    if tuple(sample.sample_id for sample in samples) != uncalibrated_store.sample_ids:
        raise ValueError("evidence calibration samples are not identically ordered")
    if uncalibrated_store.bank_hash != bank.bank_hash:
        raise ValueError("evidence calibration bank mismatch")
    if uncalibrated_store.calibration_hash != "identity":
        raise ValueError("evidence calibration must be fitted from raw uncalibrated logits")
    if "__uncalibrated__" not in uncalibrated_store.transform:
        raise ValueError("raw evidence transform is not marked uncalibrated")
    calibrated_transform = uncalibrated_store.transform.replace(
        "__uncalibrated__",
        "__train-state-mean-debias__",
        1,
    ).replace("log-margin-v1", "log-margin-v2")
    calibration = EvidenceCalibration(
        state_ids=uncalibrated_store.state_ids,
        state_logit_bias=tuple(float(value) for value in uncalibrated_store.raw_logits.mean(axis=0)),
        bank_hash=bank.bank_hash,
        model_id=uncalibrated_store.model_id,
        model_revision=uncalibrated_store.model_revision,
        text_template=uncalibrated_store.text_template,
        transform=calibrated_transform,
    )
    calibration.validate()
    return calibration


def apply_evidence_calibration(
    *,
    store: EvidenceStore,
    bank: AtomBank,
    calibration: EvidenceCalibration,
) -> EvidenceStore:
    calibration.validate()
    expected = {
        "state_ids": store.state_ids,
        "bank_hash": store.bank_hash,
        "model_id": store.model_id,
        "model_revision": store.model_revision,
        "text_template": store.text_template,
    }
    for name, value in expected.items():
        if getattr(calibration, name) != value:
            raise ValueError(f"evidence calibration mismatch: {name}")
    if bank.bank_hash != store.bank_hash:
        raise ValueError("evidence store and bank mismatch")
    probabilities, signed_margins, ordinal, applicability = _derive_group_arrays(
        raw_logits=store.raw_logits,
        bank=bank,
        state_group_array=store.state_group_index,
        state_logit_bias=np.asarray(calibration.state_logit_bias, dtype=np.float64),
    )
    result = EvidenceStore(
        sample_ids=store.sample_ids,
        group_ids=store.group_ids,
        state_ids=store.state_ids,
        state_group_index=store.state_group_index,
        raw_logits=store.raw_logits,
        state_probabilities=np.asarray(probabilities, dtype=np.float32),
        signed_margins=np.asarray(signed_margins, dtype=np.float32),
        ordinal_scores=np.asarray(ordinal, dtype=np.float32),
        applicability_scores=np.asarray(applicability, dtype=np.float32),
        bank_hash=store.bank_hash,
        model_id=store.model_id,
        model_revision=store.model_revision,
        calibration_hash=calibration.calibration_hash,
        text_template=store.text_template,
        transform=calibration.transform,
    )
    result.validate()
    return result


def align_stores(left: EvidenceStore, right: EvidenceStore) -> None:
    left.validate()
    right.validate()
    fields = (
        "group_ids",
        "state_ids",
        "bank_hash",
        "model_id",
        "model_revision",
        "calibration_hash",
        "text_template",
        "transform",
    )
    for field in fields:
        if getattr(left, field) != getattr(right, field):
            raise ValueError(f"evidence stores disagree on {field}")
    if not np.array_equal(left.state_group_index, right.state_group_index):
        raise ValueError("evidence stores disagree on state_group_index")


def group_similarity_matrix(
    query_store: EvidenceStore,
    candidate_store: EvidenceStore,
    group_index: int,
    query_indices: Sequence[int] | None = None,
    validate_alignment: bool = True,
) -> np.ndarray:
    if validate_alignment:
        align_stores(query_store, candidate_store)
    indices = query_store.state_slice(group_index)
    query_probabilities = query_store.state_probabilities[:, indices]
    if query_indices is not None:
        query_probabilities = query_probabilities[np.asarray(query_indices, dtype=np.int64)]
    candidate_probabilities = candidate_store.state_probabilities[:, indices]
    # One minus total-variation distance is a bounded, fixed-scale similarity.
    differences = np.abs(query_probabilities[:, None, :] - candidate_probabilities[None, :, :])
    return 1.0 - 0.5 * differences.sum(axis=2)


def make_response_fingerprints(
    store: EvidenceStore,
    pair_count: int = 4096,
    seed: int = 42,
) -> np.ndarray:
    if len(store.sample_ids) < 2:
        raise ValueError("response fingerprints need at least two images")
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(store.sample_ids), size=pair_count)
    right = rng.integers(0, len(store.sample_ids), size=pair_count)
    same = left == right
    right[same] = (right[same] + 1) % len(store.sample_ids)
    fingerprints = np.empty((len(store.group_ids), pair_count), dtype=np.float32)
    for group_index in range(len(store.group_ids)):
        probabilities = store.group_probability(group_index)
        fingerprints[group_index] = 1.0 - 0.5 * np.abs(probabilities[left] - probabilities[right]).sum(axis=1)
    return fingerprints

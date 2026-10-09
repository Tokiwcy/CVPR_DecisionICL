from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .evidence import EmbeddingStore
from .retrieval import RetrievalResult, global_similarity_matrix


@dataclass(frozen=True)
class BaselineProvenance:
    name: str
    status: str
    source_url: str
    notes: str


BASELINE_PROVENANCE = (
    BaselineProvenance(
        "RICES",
        "implemented",
        "https://github.com/mlfoundations/open_flamingo",
        "Full-pool image-image cosine KNN with deterministic lexical-ID ties.",
    ),
    BaselineProvenance(
        "MMICES-question-only",
        "implemented",
        "https://openaccess.thecvf.com/content/WACV2025/html/Chen_Can_Multimodal_Large_Language_Models_Truly_Perform_Multimodal_In-Context_Learning_WACV_2025_paper.html",
        "Image shortlist followed by question-text similarity; fixed-question ties retain RICES rank.",
    ),
    BaselineProvenance(
        "MUIER-classification-adapted",
        "implemented",
        "https://aclanthology.org/2024.emnlp-main.305/",
        "Image-image plus query-image to known TRAIN-label text similarity; adaptation is disclosed.",
    ),
    BaselineProvenance(
        "CIRCLES",
        "external_official_adapter_required",
        "https://github.com/gzxiong/CIRCLES",
        "Official method uses correlational plus composed attribute-guided retrieval and a different native demo budget.",
    ),
    BaselineProvenance(
        "GRIP",
        "blocked_no_official_code",
        "https://arxiv.org/abs/2606.12744",
        "Paper states that code will be released upon acceptance; no approximation is reported as GRIP.",
    ),
)


def rices(
    *,
    query_embeddings: EmbeddingStore,
    candidate_embeddings: EmbeddingStore,
    shots: int,
) -> RetrievalResult:
    if tuple(sorted(candidate_embeddings.sample_ids)) != candidate_embeddings.sample_ids:
        raise ValueError("candidate embeddings must be lexically ordered by sample_id")
    scores = global_similarity_matrix(query_embeddings, candidate_embeddings)
    rows: list[tuple[str, ...]] = []
    row_scores: list[tuple[float, ...]] = []
    candidate_ids = np.asarray(candidate_embeddings.sample_ids)
    for row in scores:
        ranking = np.argsort(-row, kind="stable")[:shots]
        rows.append(tuple(str(candidate_ids[index]) for index in ranking))
        row_scores.append(tuple(float(row[index]) for index in ranking))
    return RetrievalResult(query_embeddings.sample_ids, tuple(rows), tuple(row_scores))


def mmices_question_only(
    *,
    query_embeddings: EmbeddingStore,
    candidate_embeddings: EmbeddingStore,
    query_questions: Sequence[str],
    candidate_questions: Sequence[str],
    text_encoder: object,
    candidate_k: int,
    shots: int,
) -> RetrievalResult:
    if len(query_questions) != len(query_embeddings.sample_ids):
        raise ValueError("query questions do not align")
    if len(candidate_questions) != len(candidate_embeddings.sample_ids):
        raise ValueError("candidate questions do not align")
    if candidate_k < shots or candidate_k > len(candidate_embeddings.sample_ids):
        raise ValueError("MMICES candidate_k must cover shots and fit the train pool")
    image_scores = global_similarity_matrix(query_embeddings, candidate_embeddings)
    query_text = text_encoder.encode_texts(query_questions)
    candidate_text = text_encoder.encode_texts(candidate_questions)
    text_scores = query_text @ candidate_text.T
    candidate_ids = np.asarray(candidate_embeddings.sample_ids)
    selected: list[tuple[str, ...]] = []
    selected_scores: list[tuple[float, ...]] = []
    for query_index in range(len(query_questions)):
        image_rank = np.argsort(-image_scores[query_index], kind="stable")[:candidate_k]
        # Stable sorting preserves the image rank when fixed questions tie.
        rerank = np.argsort(-text_scores[query_index, image_rank], kind="stable")[:shots]
        final = image_rank[rerank]
        selected.append(tuple(str(candidate_ids[index]) for index in final))
        selected_scores.append(tuple(float(text_scores[query_index, index]) for index in final))
    return RetrievalResult(query_embeddings.sample_ids, tuple(selected), tuple(selected_scores))


def muier_classification_adapted(
    *,
    query_embeddings: EmbeddingStore,
    candidate_embeddings: EmbeddingStore,
    candidate_label_texts: Sequence[str],
    text_encoder: object,
    shots: int,
) -> RetrievalResult:
    if len(candidate_label_texts) != len(candidate_embeddings.sample_ids):
        raise ValueError("candidate label texts do not align")
    image_image = 2.0 * global_similarity_matrix(query_embeddings, candidate_embeddings) - 1.0
    label_text = text_encoder.encode_texts(candidate_label_texts)
    image_text = query_embeddings.embeddings @ label_text.T
    combined = image_image + image_text
    candidate_ids = np.asarray(candidate_embeddings.sample_ids)
    selected: list[tuple[str, ...]] = []
    selected_scores: list[tuple[float, ...]] = []
    for row in combined:
        ranking = np.argsort(-row, kind="stable")[:shots]
        selected.append(tuple(str(candidate_ids[index]) for index in ranking))
        selected_scores.append(tuple(float(row[index]) for index in ranking))
    return RetrievalResult(query_embeddings.sample_ids, tuple(selected), tuple(selected_scores))

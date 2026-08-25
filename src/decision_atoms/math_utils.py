from __future__ import annotations

import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return 0.0
    rx, ry = rankdata(x[keep]), rankdata(y[keep])
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def binary_auc(scores: np.ndarray, targets: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=bool)
    positive = int(targets.sum())
    negative = int((~targets).sum())
    if positive == 0 or negative == 0:
        return 0.5
    ranks = rankdata(scores)
    rank_sum = float(ranks[targets].sum())
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return normalize_rows(left) @ normalize_rows(right).T

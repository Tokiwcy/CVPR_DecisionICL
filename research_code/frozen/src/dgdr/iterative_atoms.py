"""CPU-only primitives for train-internal iterative contrast discovery.

No filesystem, VLM, validation manifest, or test outcome is accessed here.
Scoped new groups change candidate scores; they never truncate the pool.
"""
from __future__ import annotations
import numpy as np
from .atom_label import fit_nonnegative_residual, score_classes


def split_train_roles(samples, seed=42):
    if any(s.split != "train" for s in samples):
        raise ValueError("iterative discovery/audit roles require train samples only")
    labels = np.asarray([s.label for s in samples])
    rng = np.random.default_rng(seed)
    discover, audit = [], []
    for label in sorted(set(labels)):
        ids = np.flatnonzero(labels == label)
        rng.shuffle(ids)
        cut = len(ids) // 2
        if cut < 6 or len(ids) - cut < 4:
            raise ValueError("need at least six discovery and four audit images per class")
        discover.extend(ids[:cut].tolist())
        audit.extend(ids[cut:].tolist())
    return np.asarray(sorted(discover)), np.asarray(sorted(audit))


def prototype_tv(probabilities):
    """Equal group mean TV; each input is class x state probabilities."""
    if not probabilities:
        raise ValueError("no prototype groups")
    c = len(probabilities[0])
    result = np.zeros((c, c), dtype=np.float64)
    for p in probabilities:
        p = np.asarray(p, dtype=np.float64)
        if p.ndim != 2 or len(p) != c or not np.allclose(p.sum(1), 1):
            raise ValueError("invalid class probability prototype")
        result += 0.5 * np.abs(p[:, None] - p[None, :]).sum(2)
    return result / len(probabilities)


def select_pairs(predicted, truth, probabilities, seen=(), budget=5):
    """Train-audit error rate times atom proximity; no test-driven pair list."""
    d = prototype_tv(probabilities)
    c = len(d)
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    if predicted.shape != truth.shape or predicted.ndim != 1:
        raise ValueError("query alignment mismatch")
    if np.any(truth < 0) or np.any(truth >= c) or np.any(predicted < 0) or np.any(predicted >= c):
        raise ValueError("invalid class index")
    confusion = np.zeros((c, c), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    counts = np.bincount(truth, minlength=c).clip(min=1)
    omitted = {tuple(sorted(pair)) for pair in seen}
    pairs = []
    for a in range(c):
        for b in range(a + 1, c):
            errors = int(confusion[a, b] + confusion[b, a])
            if not errors or (a, b) in omitted:
                continue
            rate = 0.5 * (confusion[a, b] / counts[a] + confusion[b, a] / counts[b])
            pairs.append(dict(pair=[a, b], errors=errors, rate=float(rate),
                              atom_tv=float(d[a, b]), priority=float(rate * (1-d[a, b]))))
    return sorted(pairs, key=lambda r: (-r["priority"], -r["errors"], r["atom_tv"], r["pair"]))[:budget]


def scope_features(standardized, scopes):
    """Input query x ALL classes x (MUIER + groups), scopes=None or class indices.

    Standardization is the existing across-all-classes transform, performed before
    masking. Outside a group's fixed discovery pair, its additive contribution is
    zero. The baseline and every candidate class remain available.
    """
    x = np.asarray(standardized, dtype=np.float32).copy()
    if x.ndim != 3 or x.shape[2] != len(scopes) + 1:
        raise ValueError("scope/feature contract mismatch")
    for g, scope in enumerate(scopes, start=1):
        if scope is None:
            continue
        scope = tuple(scope)
        if len(scope) != 2 or len(set(scope)) != 2 or min(scope) < 0 or max(scope) >= x.shape[1]:
            raise ValueError("new groups require a fixed two-class support")
        outside = np.ones(x.shape[1], dtype=bool)
        outside[list(scope)] = False
        x[:, outside, g] = 0
    return x


def equal_weights(feature_count, alpha):
    if feature_count < 2:
        raise ValueError("at least one atom group required")
    w = np.full(feature_count, (1-alpha)/(feature_count-1), dtype=np.float32)
    w[0] = alpha
    return w


def calibrate(features, truth, folds, alphas=(0., .1, .2, .35, .5, .65, .8, .9, 1.), *, include_residual=True):
    """Independently fit every supplied exact bank/scope on supplied calibration rows.

    For discovery this is train-audit cross-fitting. After banks are frozen this
    same function is called afresh on full validation; coefficients are never
    passed between the calls.
    """
    x, y, folds = np.asarray(features), np.asarray(truth), np.asarray(folds)
    if x.ndim != 3 or len(x) != len(y) or len(y) != len(folds):
        raise ValueError("calibration alignment mismatch")
    if len(set(folds)) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("need finite features and multiple folds")
    def fit(rows):
        # Fixed tie order favors baseline mass, not test results.
        alpha = max(alphas, key=lambda a: (
            float(np.mean(score_classes(x[rows], equal_weights(x.shape[2], a)).argmax(1) == y[rows])), a))
        residual = fit_nonnegative_residual(x, y, rows) if include_residual else None
        return equal_weights(x.shape[2], alpha), residual
    base = x[:, :, 0].argmax(1)
    eq, residual = np.empty_like(y), np.empty_like(y)
    records = []
    for fold in sorted(set(folds)):
        train = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        w, r = fit(train)
        eq[held] = score_classes(x[held], w).argmax(1)
        if r is not None:
            residual[held] = score_classes(x[held], r).argmax(1)
        records.append(dict(fold=int(fold), equal_weights=w.tolist(),
                            residual_weights=None if r is None else r.tolist()))
    w, r = fit(np.arange(len(y)))
    passed = float(np.mean(eq == y)) > float(np.mean(base == y)) + 1e-12
    predictions=dict(muier=base, equal_raw=eq, deployed=eq if passed else base)
    if include_residual:
        predictions['residual_raw']=residual
    return dict(predictions=predictions,
                outer_folds=records, full_equal_weights=w, full_residual_weights=r,
                fallback_to_muier=not passed)


def incremental_accept(before, after, truth, pair):
    """Fixed train-audit rule: positive pair rescue and no net global harm."""
    before, after, truth = map(np.asarray, (before, after, truth))
    if before.shape != truth.shape or after.shape != truth.shape:
        raise ValueError("screen alignment mismatch")
    selected = np.isin(truth, pair)
    delta = (after == truth).astype(int) - (before == truth).astype(int)
    return bool(selected.any() and delta[selected].sum() > 0 and delta.sum() >= 0)

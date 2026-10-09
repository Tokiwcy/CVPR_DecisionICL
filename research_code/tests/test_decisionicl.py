import numpy as np

from decisionicl import (
    calibrate_decisionicl,
    class_factor_prototypes,
    class_features,
    factor_class_similarities,
    select_decisionicl_demonstrations,
)


def test_full_pool_class_selection_can_retrieve_outside_parent_top_two():
    train_ids = ("a1", "a2", "b1", "b2")
    train_labels = ("a", "a", "b", "b")
    # Global parent top two are a1, a2; DecisionICL selects class b.
    parent = np.asarray([[0.99, 0.98, 0.60, 0.55]], dtype=np.float32)
    features = np.asarray([[[0.0, -1.0], [0.0, 1.0]]], dtype=np.float32)
    chosen = select_decisionicl_demonstrations(
        parent, train_ids, train_labels, ("a", "b"), features,
        np.asarray([0.2, 0.8]), shots=2,
    )
    assert chosen == (("b1", "b2"),)


def test_factor_probabilities_class_features_and_validation_fit():
    labels, prototypes = class_factor_prototypes(
        ("a", "a", "b", "b"),
        (np.asarray([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]]),),
    )
    query = (np.asarray([[0.85, 0.15], [0.15, 0.85]]),)
    similarity = factor_class_similarities(query, prototypes)
    features = class_features(
        np.asarray([[0.5, 0.4, 0.8, 0.7], [0.8, 0.7, 0.5, 0.4]]),
        ("a", "a", "b", "b"), labels, similarity,
    )
    assert similarity.shape == (2, 2, 1)
    assert similarity[0, 0, 0] > similarity[0, 1, 0]
    assert features.shape == (2, 2, 2)
    repeated = np.tile(features, (5, 1, 1))
    truth = np.tile(np.asarray([0, 1]), 5)
    folds = np.repeat(np.arange(5), 2)
    fit = calibrate_decisionicl(repeated, truth, folds)
    assert fit.parent_mass in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
    assert fit.oof_accuracy == 1.0


def test_rejects_incomplete_validation_folds():
    features = np.zeros((5, 2, 2), dtype=np.float32)
    try:
        calibrate_decisionicl(features, [0] * 5, [0] * 5)
    except ValueError as error:
        assert "five" in str(error)
    else:
        raise AssertionError("calibration accepted a one-fold split")

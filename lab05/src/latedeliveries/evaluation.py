"""Evaluation metrics and cost functions for late deliveries."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_cost(y_true, flagged) -> float:
    """Compute total cost based on QuickBite's business rule:
    $4 for every order flagged (voucher + apology),
    $15 for every late order that was NOT flagged (refund + lost custom).
    """
    y_true = np.asarray(y_true)
    flagged = np.asarray(flagged, dtype=bool)

    vouchers = np.sum(flagged)
    missed_lates = np.sum((y_true == 1) & (~flagged))

    return float(4 * vouchers + 15 * missed_lates)


def find_optimal_threshold(y_true, y_prob, thresholds=None) -> float:
    """Find the threshold from 0.05 to 0.95 that minimizes total cost."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)

    best_threshold = 0.5
    best_cost = float("inf")

    for th in thresholds:
        flagged = y_prob >= th
        cost = compute_cost(y_true, flagged)
        if cost < best_cost:
            best_cost = cost
            best_threshold = th

    return float(best_threshold)


def evaluate_predictions(y_true, y_pred, y_prob=None) -> dict:
    """Compute standard classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.0

    return metrics

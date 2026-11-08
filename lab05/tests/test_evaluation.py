"""Tests for cost function, thresholds, and probabilities."""

import numpy as np

from latedeliveries.evaluation import compute_cost, find_optimal_threshold


def test_cost_of_a_small_example():
    """Test cost calculation rule on a small handmade example."""
    y_true = [1, 0, 0, 1, 0]
    flagged = [True, True, False, False, False]

    # 2 vouchers (2 * 4 = 8) + 1 missed late order (1 * 15 = 15) = 23
    assert compute_cost(y_true, flagged) == (2 * 4 + 1 * 15)


def test_threshold_selection_extremes():
    """Test that threshold function can handle extreme policies."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    threshold = find_optimal_threshold(y_true, y_prob)
    assert 0.05 <= threshold <= 0.95

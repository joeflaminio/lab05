"""Tests for feature selection, pipeline preprocessing, and splitting."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from latedeliveries.pipeline import build_pipeline, get_features_and_target


def test_feature_list_excludes_leakage(sample_data):
    """Ensure data leakage columns are excluded from features."""
    X, y = get_features_and_target(sample_data)

    forbidden_cols = ["prep_minutes", "customer_rating", "late", "order_id"]
    for col in forbidden_cols:
        assert col not in X.columns, f"Forbidden column '{col}' found in features!"


@pytest.mark.parametrize("test_size", [0.2, 0.25, 0.3])
def test_stratified_split(sample_data, test_size):
    """Test that train-test split is stratified and repeatable."""
    X, y = get_features_and_target(sample_data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    train_late_rate = y_train.mean()
    test_late_rate = y_test.mean()

    assert abs(train_late_rate - test_late_rate) < 0.05, (
        "Split is not properly stratified!"
    )


def test_pipeline_handles_missing_and_unseen(sample_data):
    """Test that pipeline successfully handles missing values and unseen categories."""
    X, y = get_features_and_target(sample_data)

    pipeline = build_pipeline(LogisticRegression())
    pipeline.fit(X, y)

    unseen_df = pd.DataFrame(
        {
            "distance_km": [None],
            "items": [3],
            "hour": [15],
            "weather": ["hurricane_unseen"],
            "day_type": ["weekday"],
            "vehicle": ["bike"],
            "courier_months": [12],
        }
    )

    preds = pipeline.predict(unseen_df)
    assert len(preds) == 1

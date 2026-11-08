"""Tests for model serialization and reloading."""

import joblib
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import build_pipeline, get_features_and_target


def test_saved_model_prediction(sample_data, tmp_path):
    """Test saving and reloading model via joblib produces identical predictions."""
    X, y = get_features_and_target(sample_data)

    pipeline = build_pipeline(LogisticRegression())
    pipeline.fit(X, y)

    model_file = tmp_path / "late_orders.joblib"
    joblib.dump(pipeline, model_file)

    loaded_model = joblib.load(model_file)

    orig_preds = pipeline.predict(X.head(10))
    loaded_preds = loaded_model.predict(X.head(10))

    assert (orig_preds == loaded_preds).all()

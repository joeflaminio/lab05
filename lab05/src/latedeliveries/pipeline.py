"""Preprocessing and feature engineering pipeline for late deliveries."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_features_and_target(df: pd.DataFrame):
    """Separate features from target, avoiding data leakage columns."""
    feature_cols = [
        "distance_km",
        "items",
        "hour",
        "weather",
        "day_type",
        "vehicle",
        "courier_months",
    ]
    X = df[feature_cols]
    y = df["late"]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Construct a ColumnTransformer for scaling numeric and encoding categorical features."""
    numeric_features = ["distance_km", "items", "hour", "courier_months"]
    categorical_features = ["weather", "day_type", "vehicle"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor

"""Pipeline construction and feature selection for late deliveries."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Define feature groups explicitly
NUMERIC_FEATURES = ["distance_km", "items", "hour", "courier_months"]
CATEGORICAL_FEATURES = ["weather", "day_type", "vehicle"]
LEAKAGE_COLUMNS = ["prep_minutes", "customer_rating", "late", "order_id"]


def get_features_and_target(df: pd.DataFrame):
    """Extract features (X) and target (y), ensuring leakage columns are excluded."""
    # Drop target and any potential data leakage columns if present
    drop_cols = [c for c in LEAKAGE_COLUMNS + ["late"] if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df["late"]

    # Keep only defined feature columns
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = X[[c for c in feature_cols if c in X.columns]]

    return X, y


def build_pipeline(model) -> Pipeline:
    """Take a scikit-learn model and return a
    complete Pipeline with the preprocessor.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

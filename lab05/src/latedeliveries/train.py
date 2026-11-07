"""Main training script for Lab 05: Late Deliveries prediction."""

import json
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.base import clone

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_val_score,
    cross_val_predict,
    GridSearchCV,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from latedeliveries.data import make_orders
from latedeliveries.pipeline import get_features_and_target, build_pipeline
from latedeliveries.evaluation import (
    compute_cost,
    find_optimal_threshold,
    evaluate_predictions,
)

RANDOM_STATE = 42


def main():
    # 1. Load data and split
    df = make_orders(seed=7)
    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 2. Model Comparison via 5-fold CV
    models = {
        "Baseline_Dummy": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "LogReg_Balanced": LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = []

    print("Running cross-validation model comparison...")
    for name, model in models.items():
        pipeline = build_pipeline(model)
        
        # Scoring metrics
        acc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        prec_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="precision")
        rec_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="recall")
        f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
        roc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

        cv_results.append({
            "model": name,
            "accuracy_mean": float(np.mean(acc_scores)),
            "accuracy_std": float(np.std(acc_scores)),
            "precision_mean": float(np.mean(prec_scores)),
            "precision_std": float(np.std(prec_scores)),
            "recall_mean": float(np.mean(rec_scores)),
            "recall_std": float(np.std(rec_scores)),
            "f1_mean": float(np.mean(f1_scores)),
            "f1_std": float(np.std(f1_scores)),
            "roc_auc_mean": float(np.mean(roc_scores)),
            "roc_auc_std": float(np.std(roc_scores)),
        })

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv("reports/cv_results.csv", index=False)
    print("Saved CV results to reports/cv_results.csv")

    # 3. Choose model & Tune with GridSearchCV (e.g., Logistic Regression or Random Forest)
    chosen_base_model = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)
    chosen_pipeline = build_pipeline(chosen_base_model)

    param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__solver": ["lbfgs", "liblinear"],
    }

    print("Running GridSearchCV...")
    grid_search = GridSearchCV(
        chosen_pipeline,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)

    # 4. Threshold Selection using out-of-fold probabilities
    print("Selecting optimal decision threshold based on cost...")
    oof_probs = cross_val_predict(
        clone(best_pipeline), X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]

    optimal_threshold = find_optimal_threshold(y_train, oof_probs)
    print(f"Optimal decision threshold: {optimal_threshold}")

    # 5. Score Test Set Once
    print("Evaluating final model on test set...")
    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= optimal_threshold).astype(int)

    test_metrics = evaluate_predictions(y_test, test_pred, test_probs)
    cm = confusion_matrix(y_test, test_pred).tolist()

    # Calculate costs on test set for different policies
    flag_none = np.zeros_like(y_test)
    flag_all = np.ones_like(y_test)
    
    cost_none = compute_cost(y_test, flag_none)
    cost_all = compute_cost(y_test, flag_all)
    cost_model = compute_cost(y_test, test_pred)

    final_metrics = {
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "optimal_threshold": optimal_threshold,
        "costs": {
            "flag_none": cost_none,
            "flag_all": cost_all,
            "model_policy": cost_model,
        },
    }

    with open("reports/metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
    print("Saved test metrics to reports/metrics.json")

    # 6. Save Model and Metadata
    joblib.dump(best_pipeline, "models/late_orders.joblib")
    print("Saved final pipeline to models/late_orders.joblib")

    model_info = {
        "sklearn_version": __import__("sklearn").__version__,
        "seed": 7,
        "random_state": RANDOM_STATE,
        "threshold": optimal_threshold,
        "best_params": grid_search.best_params_,
        "feature_lists": {
            "numeric": ["distance_km", "items", "hour", "courier_months"],
            "categorical": ["weather", "day_type", "vehicle"],
        },
    }

    with open("models/model_info.json", "w") as f:
        json.dump(model_info, f, indent=4)
    print("Saved model info to models/model_info.json")


if __name__ == "__main__":
    main()

"""End-to-end training pipeline for the Mobile App Rating regression project.

This script performs the following steps:

1. Loads the curated Apple App Store mindfulness dataset.
2. Cleans and engineers features for regression tasks.
3. Trains Linear Regression, Decision Tree, Random Forest, and SGD (gradient descent) models.
4. Logs metrics, plots loss curves & scatter plots, and saves the best-performing model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent.parent / "data" / "mindful_app_store.csv"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

RANDOM_STATE = 42


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=[
        "avg_rating",
        "price",
        "rating_count",
        "size_mb",
        "language_count",
        "age_days",
        "update_recency_days",
        "primary_genre",
        "content_rating",
    ])

    df = df[df["avg_rating"].between(1, 5)]
    df = df[df["rating_count"] > 0]

    def parse_min_ios(value: str | float | int | None) -> float | None:
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        stripped = str(value).split(" ")[0]
        try:
            return float(stripped)
        except ValueError:
            return None

    df["min_ios_numeric"] = df["min_ios"].apply(parse_min_ios)
    df["min_ios_numeric"] = df["min_ios_numeric"].fillna(
        df["min_ios_numeric"].median()
    )

    df["log_rating_count"] = np.log1p(df["rating_count"])

    # Drop duplicates by app name to reduce leakage when terms overlap
    df = df.drop_duplicates(subset=["app_name"])

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "price",
        "rating_count",
        "size_mb",
        "language_count",
        "has_iap",
        "has_support_url",
        "is_game_center",
        "age_days",
        "update_recency_days",
        "min_ios_numeric",
    ]
    categorical_features = ["primary_genre", "content_rating"]

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    pipeline.fit(X_train, y_train)
    preds_train = pipeline.predict(X_train)
    preds_test = pipeline.predict(X_test)

    metrics = {
        "model": name,
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, preds_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, preds_test))),
        "train_mae": float(mean_absolute_error(y_train, preds_train)),
        "test_mae": float(mean_absolute_error(y_test, preds_test)),
        "train_r2": float(r2_score(y_train, preds_train)),
        "test_r2": float(r2_score(y_test, preds_test)),
    }
    return metrics


def train_gradient_descent(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    epochs: int = 60,
    eta0: float = 0.01,
) -> Tuple[Pipeline, List[float], List[float]]:
    sgd = SGDRegressor(
        loss="squared_error",
        penalty=None,
        max_iter=1,
        learning_rate="constant",
        eta0=eta0,
        random_state=RANDOM_STATE,
        warm_start=True,
        tol=None,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", sgd)])

    train_losses: List[float] = []
    test_losses: List[float] = []

    for _ in range(epochs):
        pipeline.fit(X_train, y_train)
        preds_train = pipeline.predict(X_train)
        preds_test = pipeline.predict(X_test)
        train_losses.append(mean_squared_error(y_train, preds_train))
        test_losses.append(mean_squared_error(y_test, preds_test))

    return pipeline, train_losses, test_losses


def plot_loss_curves(train_losses: List[float], test_losses: List[float]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train MSE", linewidth=2)
    plt.plot(test_losses, label="Test MSE", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("SGD Gradient Descent Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "loss_curve.png", dpi=200)
    plt.close()


def plot_scatter_with_line(df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    x = df["log_rating_count"].values
    y = df["avg_rating"].values
    slope, intercept = np.polyfit(x, y, 1)
    x_sorted = np.linspace(x.min(), x.max(), 200)
    y_pred_line = slope * x_sorted + intercept

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.4, label="Actual Ratings", color="steelblue")
    plt.plot(x_sorted, y_pred_line, color="darkorange", linewidth=2.5, label="Best Fit Line")
    plt.xlabel("log(1 + Rating Count)")
    plt.ylabel("Average User Rating")
    plt.title("Scatter Plot with Regression Line")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "rating_line_fit.png", dpi=200)
    plt.close()


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)
    df = engineer_features(df)

    features = df.drop(columns=["avg_rating", "app_name"])
    target = df["avg_rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=pd.qcut(target, q=4, duplicates="drop"),
    )

    preprocessor = build_preprocessor()

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    evaluation_results: List[Dict[str, float]] = []
    trained_pipelines: Dict[str, Pipeline] = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        metrics = evaluate_model(name, pipeline, X_train, y_train, X_test, y_test)
        evaluation_results.append(metrics)
        trained_pipelines[name] = pipeline

    sgd_pipeline, train_losses, test_losses = train_gradient_descent(
        preprocessor, X_train, y_train, X_test, y_test, epochs=80, eta0=0.01
    )
    metrics_sgd = evaluate_model(
        "SGDRegressor",
        sgd_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    evaluation_results.append(metrics_sgd)
    trained_pipelines["SGDRegressor"] = sgd_pipeline

    plot_loss_curves(train_losses, test_losses)
    plot_scatter_with_line(df)

    metrics_path = REPORTS_DIR / "model_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)

    best_model = min(evaluation_results, key=lambda m: m["test_rmse"])
    best_name = best_model["model"]
    joblib.dump(
        {
            "pipeline": trained_pipelines[best_name],
            "metrics": best_model,
            "feature_columns": list(features.columns),
        },
        MODELS_DIR / "best_model.joblib",
    )

    print("Training complete. Best model:", best_model)
    print("Metrics saved to", metrics_path)


if __name__ == "__main__":
    main()
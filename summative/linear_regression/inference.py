"""Utility functions for loading the trained model and generating predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"


REQUIRED_INPUT_FIELDS = [
    "price",
    "rating_count",
    "size_mb",
    "primary_genre",
    "content_rating",
    "language_count",
    "has_iap",
    "has_support_url",
    "min_ios",
    "is_game_center",
    "age_days",
    "update_recency_days",
]


def _parse_min_ios(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).split(" ")[0]
    try:
        return float(text)
    except ValueError:
        return 0.0


def _add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["min_ios_numeric"] = df["min_ios"].apply(_parse_min_ios)
    df["log_rating_count"] = np.log1p(df["rating_count"].clip(lower=0))
    return df


def _ensure_columns(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    for column in feature_columns:
        if column not in df.columns:
            df[column] = 0
    return df.reindex(columns=feature_columns, fill_value=0)


def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Best model artifact not found. Please run train_models.py first."
        )
    return joblib.load(MODEL_PATH)


def predict(payload: Dict[str, Any]) -> float:
    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]

    df = pd.DataFrame([payload])
    df = _add_engineered_columns(df)
    df = _ensure_columns(df, feature_columns)

    prediction = pipeline.predict(df)[0]
    return float(prediction)


def demo_prediction() -> float:
    artifact = load_artifact()
    feature_columns = artifact["feature_columns"]
    sample_payload = {col: 0 for col in REQUIRED_INPUT_FIELDS}
    # Provide sensible defaults
    sample_payload.update(
        {
            "price": 0.0,
            "rating_count": 1200,
            "size_mb": 150.0,
            "primary_genre": "Health & Fitness",
            "content_rating": "4+",
            "language_count": 8,
            "has_iap": 1,
            "has_support_url": 1,
            "min_ios": "13.0",
            "is_game_center": 0,
            "age_days": 1800,
            "update_recency_days": 45,
        }
    )
    df = pd.DataFrame([sample_payload])
    df = _add_engineered_columns(df)
    df = _ensure_columns(df, feature_columns)
    prediction = artifact["pipeline"].predict(df)[0]
    return float(prediction)


__all__ = ["predict", "load_artifact", "REQUIRED_INPUT_FIELDS", "demo_prediction"]
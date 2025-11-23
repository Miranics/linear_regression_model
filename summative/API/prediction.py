"""FastAPI service that exposes the best-performing regression model via /predict."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

API_DIR = Path(__file__).resolve().parent
LINEAR_REGRESSION_DIR = API_DIR.parent / "linear_regression"
sys.path.append(str(LINEAR_REGRESSION_DIR))

try:
    from inference import load_artifact, predict
except ModuleNotFoundError as exc:  # pragma: no cover - import helper
    raise RuntimeError("Unable to import inference utilities. Check PYTHONPATH.") from exc


GENRE_CHOICES = [
    "Book",
    "Business",
    "Education",
    "Entertainment",
    "Finance",
    "Food & Drink",
    "Games",
    "Graphics & Design",
    "Health & Fitness",
    "Lifestyle",
    "Medical",
    "Music",
    "News",
    "Photo & Video",
    "Productivity",
    "Reference",
    "Shopping",
    "Utilities",
]

CONTENT_RATINGS = ["4+", "9+", "12+", "17+"]


class PredictionRequest(BaseModel):
    price: float = Field(ge=0, le=100, description="App Store price in USD")
    rating_count: int = Field(ge=1, le=4_000_000, description="Total user ratings")
    size_mb: float = Field(ge=1, le=2_000, description="Binary size in megabytes")
    primary_genre: Literal[
        "Book",
        "Business",
        "Education",
        "Entertainment",
        "Finance",
        "Food & Drink",
        "Games",
        "Graphics & Design",
        "Health & Fitness",
        "Lifestyle",
        "Medical",
        "Music",
        "News",
        "Photo & Video",
        "Productivity",
        "Reference",
        "Shopping",
        "Utilities",
    ]
    content_rating: Literal["4+", "9+", "12+", "17+"]
    language_count: int = Field(ge=1, le=60, description="Number of supported languages")
    has_iap: bool
    has_support_url: bool
    min_ios: str = Field(pattern=r"^\d{1,2}(\.\d)?$", description="Minimum iOS version, e.g. '13.0'")
    is_game_center: bool
    age_days: int = Field(ge=0, le=7_000, description="Days since original release")
    update_recency_days: int = Field(ge=0, le=5_000, description="Days since last update")


class PredictionResponse(BaseModel):
    predicted_rating: float = Field(description="Forecasted average user rating")
    model_name: str
    test_rmse: float
    test_mae: float
    test_r2: float


app = FastAPI(
    title="CalmPulse Rating Regressor",
    description="Predict the average App Store rating for wellbeing apps.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="Service status")
def root() -> dict:
    artifact = load_artifact()
    metrics = artifact.get("metrics", {})
    return {
        "mission": "Forecast App Store ratings for the CalmPulse wellbeing initiative.",
        "best_model": metrics.get("model", "unknown"),
        "test_rmse": metrics.get("test_rmse"),
        "test_mae": metrics.get("test_mae"),
        "test_r2": metrics.get("test_r2"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_rating(request: PredictionRequest) -> PredictionResponse:
    payload = request.model_dump()
    # Convert booleans serialized as ints back to ints for inference
    payload.update(
        {
            "has_iap": int(request.has_iap),
            "has_support_url": int(request.has_support_url),
            "is_game_center": int(request.is_game_center),
        }
    )

    try:
        rating = predict(payload)
    except FileNotFoundError as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=500, detail=str(exc))

    artifact = load_artifact()
    metrics = artifact.get("metrics", {})
    return PredictionResponse(
        predicted_rating=round(rating, 3),
        model_name=metrics.get("model", "unknown"),
        test_rmse=metrics.get("test_rmse", 0.0),
        test_mae=metrics.get("test_mae", 0.0),
        test_r2=metrics.get("test_r2", 0.0),
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("prediction:app", host="0.0.0.0", port=8000, reload=False)
"""Quick utility that loads the persisted model and predicts one sample."""

from __future__ import annotations

from pathlib import Path

from inference import REQUIRED_INPUT_FIELDS, predict


def main() -> None:
    sample_payload = {
        "price": 0.0,
        "rating_count": 3500,
        "size_mb": 110.0,
        "primary_genre": "Health & Fitness",
        "content_rating": "4+",
        "language_count": 12,
        "has_iap": 1,
        "has_support_url": 1,
        "min_ios": "13.0",
        "is_game_center": 0,
        "age_days": 2200,
        "update_recency_days": 30,
    }
    missing = [field for field in REQUIRED_INPUT_FIELDS if field not in sample_payload]
    if missing:
        raise ValueError(f"Sample payload must include: {missing}")

    prediction = predict(sample_payload)
    print("Sample payload:", sample_payload)
    print(f"Predicted rating: {prediction:.3f}")


if __name__ == "__main__":
    main()
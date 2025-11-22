"""Utility to rebuild the mindful_app_store.csv dataset using the iTunes Search API."""

from __future__ import annotations

import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

SEARCH_TERMS = [
    "meditation",
    "mindfulness",
    "sleep",
    "habit",
    "focus",
    "wellness",
    "relax",
]

OUTPUT_PATH = Path(__file__).resolve().parent / "mindful_app_store.csv"


def fetch_apps(term: str) -> list[dict]:
    params = urlencode({"term": term, "entity": "software", "country": "us", "limit": 200})
    url = f"https://itunes.apple.com/search?{params}"
    with urlopen(url) as resp:
        payload = json.load(resp)
    return payload.get("results", [])


def normalize_app(app: dict) -> dict | None:
    rating = app.get("averageUserRating")
    rating_count = app.get("userRatingCount")
    if rating is None or rating_count is None:
        return None
    try:
        file_size = float(app.get("fileSizeBytes") or 0) / 1e6
    except (TypeError, ValueError):
        file_size = None

    def parse_date(value: str | None) -> datetime | None:
        if not value:
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    release_date = parse_date(app.get("releaseDate"))
    current_version = parse_date(app.get("currentVersionReleaseDate"))

    now = datetime.now(UTC)
    return {
        "app_name": app.get("trackName"),
        "price": app.get("price", 0.0),
        "avg_rating": rating,
        "rating_count": rating_count,
        "size_mb": file_size,
        "primary_genre": app.get("primaryGenreName"),
        "content_rating": app.get("contentAdvisoryRating"),
        "language_count": len(app.get("languageCodesISO2A", [])),
        "has_iap": 1 if app.get("inAppPurchases") else 0,
        "has_support_url": 1 if app.get("supportUrl") else 0,
        "min_ios": app.get("minimumOsVersion"),
        "is_game_center": 1 if app.get("isGameCenterEnabled") else 0,
        "age_days": (now - release_date).days if release_date else None,
        "update_recency_days": (now - current_version).days if current_version else None,
    }


def collect_dataset(terms: Iterable[str] = SEARCH_TERMS) -> None:
    unique_apps: dict[int, dict] = {}
    for term in terms:
        apps = fetch_apps(term)
        for app in apps:
            track_id = app.get("trackId")
            if track_id is None:
                continue
            normalized = normalize_app(app)
            if normalized is None:
                continue
            unique_apps[track_id] = normalized
        time.sleep(0.2)

    rows = list(unique_apps.values())
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    collect_dataset()

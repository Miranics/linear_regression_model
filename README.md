# CalmPulse Linear Regression Project

Forecast App Store ratings for wellbeing and focus companions so the CalmPulse product team can prioritize UX investments that keep users engaged. This repo houses the notebook, API service, and Flutter mobile client required for the assignment rubric.

## Repository layout

```
linear_regression_model/
└── summative/
    ├── linear_regression/        # Notebook, data, models, reports
    ├── API/                      # FastAPI service (prediction.py + requirements)
    └── FlutterApp/               # Single-screen Flutter UI hitting the API
```

## Task 1 – Notebook & model artifacts

- Notebook: `summative/linear_regression/multivariate.ipynb`
- Dataset: `summative/linear_regression/data/mindful_app_store.csv`
- Trained artifact: `summative/linear_regression/models/best_model.joblib`
- Reports: PNG loss plots + metrics JSON in `summative/linear_regression/reports/`

### Run locally

```bash
# From repo root
source .venv/bin/activate  # or use ./.venv/bin/python explicitly
pip install -r summative/API/requirements.txt
jupyter notebook summative/linear_regression/multivariate.ipynb
```

## Task 2 – FastAPI service

- Source: `summative/API/prediction.py`
- Requirements: `summative/API/requirements.txt`
- Live Render URL: **https://linear-regression-model-mxox.onrender.com**
  - Swagger UI: https://linear-regression-model-mxox.onrender.com/docs

### Local run

```bash
source .venv/bin/activate
pip install -r summative/API/requirements.txt
uvicorn prediction:app --host 0.0.0.0 --port 8000 --app-dir summative/API
```

### Render deployment recap

| Setting            | Value |
|--------------------|-------|
| Environment        | Python 3.12 |
| Build Command      | `pip install -r summative/API/requirements.txt` |
| Start Command      | `uvicorn prediction:app --host 0.0.0.0 --port $PORT --app-dir summative/API` |
| Env vars           | `PYTHONUNBUFFERED=1` (PORT provided by Render) |

## Task 3 – Flutter app

Directory: `summative/FlutterApp/`

- Default API base: `https://linear-regression-model-mxox.onrender.com` (override with `--dart-define API_BASE_URL=https://...`).
- Text fields mirror each input required by the Pydantic schema.
- Displays the predicted rating or validation errors from the API.

Run with:

```bash
cd summative/FlutterApp
flutter pub get
flutter run \
  --dart-define API_BASE_URL=https://linear-regression-model-mxox.onrender.com
```

## Task 4 – Demo video checklist

- Show notebook insights + model comparison (max 5 minutes total).
- Demonstrate Swagger `/predict` success & validation errors.
- Show Flutter app sending inputs and rendering the response.
- Camera on, concise mission/problem explanation.

## Submission reminders

- README must include mission sentence (done) and public API URL.
- Include YouTube demo link once recorded.
- Keep repo synced: `git add . && git commit -m "Update docs" && git push`.

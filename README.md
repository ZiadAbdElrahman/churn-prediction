# Churn Prediction System

This project implements an end-to-end machine learning pipeline for churn prediction, with **model training, evaluation, experiment tracking, and serving** in production.

It combines:
- **XGBoost classifier** with configurable hyperparameters  
- **MLflow** for experiment tracking and logging  
- **FastAPI** service for real-time predictions and training endpoint  
- **Docker** and **uv** for reproducible builds and environments  

---

## 🚀 Features

- **Training**
  - Train an XGBoost churn model with configurable hyperparameters.
  - Logs metrics (AUC-PR, logloss, classification report) and artifacts to MLflow.
  - Stores model, threshold, and metadata under `artifacts/`.

- **Serving**
  - `/train` endpoint to retrain the model on demand.
  - `/predict` endpoint for batch or single predictions.

---

## 📦 Project Structure

```
.
├── src/
│   ├── core/                     # Core ML logic
│   │   ├── train.py              # Training script
│   │   ├── training_pipeline.py  # Full training pipeline
│   │   ├── data_access.py        # Data loading & preprocessing
│   │   ├── features.py           # Feature engineering
│   │   ├── model_io.py           # Save/load model artifacts
│   │   └── __init__.py
│   │
│   ├── engine/                   # Business / model engine layer
│   │   └── controller/           # Controller logic
│   │       ├── churn_detection.py
│   │       ├── types.py
│   │       └── __init__.py
│   │
│   ├── service/                  # API service layer
│   │   ├── main.py               # FastAPI entrypoint
│   │   ├── churn_detection.py    # API endpoints
│   │   ├── dependencies.py       # Config & shared deps
│   │   ├── types.py              # Request/response schemas
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── config/
│   └── config.yaml               # Global configuration
│
├── data/                         # Raw and reference data
│   ├── put_data_here             # Placeholder
│   ├── customer_churn.json       # Full dataset (large)
│   └── customer_churn_mini.json  # Mini dataset for testing
│
├── artifacts/                    # Models, metrics, logs
├── notebooks/                    # Experimentation notebooks
│
├── .gitignore
├── .pre-commit-config.yaml       # Code quality hooks (ruff, black, etc.)
├── Dockerfile
├── Makefile                      # Reproducible commands
├── pyproject.toml                # Dependencies & build config (uv / PEP 621)
├── README.md
```

---



## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/churn-prediction.git
cd churn-prediction
```

## 📂 Dataset
Before running training or serving, you need to download the data file in the /data folder:
```bash
customer_churn.json
```

### Build and Run with Docker + Make

Build the Docker image:
```bash
make build
```

Run the API service:
```bash
make run
```

The service will be available at: http://localhost:8000


## 🏗️ Usage

### 1. Train a model
You can train via API:
```bash
curl -X POST "http://0.0.0.0:8000/api/v1/train" \
  -H "Content-Type: application/json"
```

### 2. Inference
You can run via API:
```bash
curl -X POST "http://0.0.0.0:8000/api/v1/predict" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
  "user_id": "100024",
  "date": "2018-10-08 21:04:57"
}'
```

---

## 📊 MLflow Tracking

Each training run logs:
- Metrics (PR-AUC, recall, precision)
- Parameters (XGBoost hyperparams)
- Artifacts (model, reports)

Run UI:
```bash
mlflow ui
```

---

## 🧹 Development

This repo uses **pre-commit hooks** for code quality.

Install hooks:
```bash
pre-commit install
```

Run checks:
```bash
pre-commit run --all-files
```

---

## 📌 TODO / Extensions

- Add monitoring system:
  - **Data drift** detection (PSI, KS test per feature).  
  - **Concept drift** detection (performance decay).  
  - Automated performance tracking.  
- Integrate retraining job on severe drift.  
- Integrate alerting (Slack/Email).  
- Add dashboard for monitoring visualization.  

---

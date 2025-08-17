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
├── core/
│   ├── train.py          # Training script
│   ├── model_io.py       # Save/load artifacts
│   ├── dependencies.py   # Config & shared utilities
│
├── app/
│   ├── main.py           # FastAPI entrypoint
│   ├── controller.py     # API endpoints
│
├── artifacts/            # Models, references, metrics
├── Makefile              # Reproducible commands
├── pyproject.toml        # Project dependencies (uv / PEP 621)
├── .pre-commit-config.yaml  # Code quality hooks
└── README.md
```

---

## ⚙️ Installation

Clone the repo and install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset
Before running training or serving, you need to download the data file in teh /data folder:
```bash
customer_churn.json
```


## 🏗️ Usage

### 1. Train a model
```bash
make train
```

Artifacts saved to `artifacts/`, including:
- `model.pkl` – trained model  
- `metadata.json` – metrics + threshold  

### 2. Run API service
```bash
make serve
```
API runs at `http://0.0.0.0:8000/`

Endpoints:
- `POST /train` → retrain model (saves artifacts + logs to MLflow).  
- `POST /predict` → predict churn probabilities.  

Example `curl`:

```bash
curl -X POST "http://0.0.0.0:8000/predict"   -H "Content-Type: application/json"   -d '{"user_id": 123, "feature1": 5.0, "feature2": 0.7}'
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

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier
from .model_io import save_artifacts, save_metadata
from ..dependencies import get_settings

import mlflow
from mlflow import sklearn as mlflow_sklearn

cfg = get_settings()

mlflow.set_tracking_uri(f"file:{cfg['paths']['artifacts_dir']}/mlruns")  # local dir
mlflow.set_experiment("churn_detection")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def _pick_threshold_for_recall_with_precision_floor(
    y_true, y_prob, precision_floor: float
) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec_, rec_, thr_ = prec[:-1], rec[:-1], thr
    valid = prec_ >= precision_floor
    if not np.any(valid):
        logger.warning(
            "No threshold meets precision floor=%.3f. Falling back.", precision_floor
        )
        return float(thr_[np.argmax(prec_)]) if len(thr_) else 0.5
    idx_valid = np.where(valid)[0]
    best_idx = idx_valid[np.argmax(rec_[valid])]
    return float(thr_[best_idx])


def _metrics_at_threshold(y_true, y_prob, thr):
    y_hat = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "tp": int(cm[1, 1]),
        "fp": int(cm[0, 1]),
        "tn": int(cm[0, 0]),
        "fn": int(cm[1, 0]),
        "y_hat": y_hat,
    }


def _clean_report_text(rep_str: str) -> str:
    lines = [ln.strip() for ln in rep_str.strip().splitlines() if ln.strip()]
    return "\n".join(lines)


def train_from_features(X: pd.DataFrame, y: pd.Series) -> dict:
    logger.info(
        "Starting training with %d samples and %d features", X.shape[0], X.shape[1]
    )
    cfg = get_settings()
    logger.info("Config(model): %s", cfg["model"])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    logger.info(
        "Split -> train: %d (pos=%.4f), val: %d (pos=%.4f)",
        X_tr.shape[0],
        y_tr.mean(),
        X_val.shape[0],
        y_val.mean(),
    )

    with mlflow.start_run():
        pos_rate = y_tr.mean()
        spw = cfg["model"]["scale_pos_weight"]
        if spw is None:
            spw = (1 - pos_rate) / max(pos_rate, 1e-9)
            logger.info("Auto scale_pos_weight=%.4f", spw)

        params = {
            "n_estimators": 1000,
            "max_depth": None,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 2,
            "reg_lambda": 1.0,
            "gamma": 0.0,
            "scale_pos_weight": spw,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "eval_metric": ["aucpr", "logloss"],
        }

        xgb = XGBClassifier(**params)
        xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=True)

        mlflow.log_params(params)
        mlflow.xgboost.log_model(xgb, artifact_path="xgb_model")

        # Probabilities
        y_tr_prob = xgb.predict_proba(X_tr)[:, 1]
        y_val_prob = xgb.predict_proba(X_val)[:, 1]

        # PR-AUC (val)
        val_pr_auc = float(average_precision_score(y_val, y_val_prob))

        # Threshold: maximize recall with precision floor
        precision_floor = float(cfg["model"]["precision_floor"])
        best_t = _pick_threshold_for_recall_with_precision_floor(
            y_val, y_val_prob, precision_floor
        )

        # Metrics at threshold
        train_metrics = _metrics_at_threshold(y_tr, y_tr_prob, best_t)
        val_metrics = _metrics_at_threshold(y_val, y_val_prob, best_t)

        # Classification reports
        train_report_text = _clean_report_text(
            classification_report(y_tr, train_metrics["y_hat"], digits=3)
        )
        val_report_text = _clean_report_text(
            classification_report(y_val, val_metrics["y_hat"], digits=3)
        )
        train_report_dict = classification_report(
            y_tr, train_metrics["y_hat"], digits=3, output_dict=True
        )
        val_report_dict = classification_report(
            y_val, val_metrics["y_hat"], digits=3, output_dict=True
        )

        # Log concise summary
        logger.info(
            "Train@thr=%.4f acc=%.3f prec=%.3f rec=%.3f f1=%.3f PR-AUC(val)=%.3f",
            best_t,
            train_metrics["accuracy"],
            train_metrics["precision"],
            train_metrics["recall"],
            train_metrics["f1"],
            val_pr_auc,
        )
        logger.info(
            "Val@thr=%.4f   acc=%.3f prec=%.3f rec=%.3f f1=%.3f  (tp=%d fp=%d tn=%d fn=%d)",
            best_t,
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
            val_metrics["tp"],
            val_metrics["fp"],
            val_metrics["tn"],
            val_metrics["fn"],
        )

        # Persist artifacts
        save_artifacts(xgb, best_t, X.columns)

        mlflow.log_metrics(
            {
                "val_pr_auc": val_pr_auc,
                "val_acc": val_metrics["accuracy"],
                "val_prec": val_metrics["precision"],
                "val_rec": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "train_acc": train_metrics["accuracy"],
                "train_prec": train_metrics["precision"],
                "train_rec": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
            }
        )

        # Bundle metadata
        results = {
            "threshold": float(best_t),
            "scale_pos_weight": float(spw),
            "pos_rate_train": float(pos_rate),
            "val_pr_auc": val_pr_auc,
            # Validation (numbers)
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_tp": val_metrics["tp"],
            "val_fp": val_metrics["fp"],
            "val_tn": val_metrics["tn"],
            "val_fn": val_metrics["fn"],
            # Train (numbers)
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            # Cleaned text reports
            "train_classification_report": train_report_text,
            "val_classification_report": val_report_text,
            # Machine-friendly dicts (optional but useful)
            "train_report_dict": train_report_dict,
            "val_report_dict": val_report_dict,
        }

        # Save metadata JSON
        save_metadata(results)
        logger.info("Artifacts + metadata saved.")
        return results

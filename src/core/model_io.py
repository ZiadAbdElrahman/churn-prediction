# model_io.py
from pathlib import Path
import json
from xgboost import XGBClassifier
from ..dependencies import get_settings


def _p(key: str) -> Path:
    cfg = get_settings()
    return Path(cfg["paths"][key]) if key in cfg["paths"] else Path(key)


def load_artifacts():
    model_path = _p("model_file")
    threshold_path = _p("threshold_file")
    features_path = _p("feature_list_file")

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not threshold_path.exists():
        raise FileNotFoundError(threshold_path)
    if not features_path.exists():
        raise FileNotFoundError(features_path)

    model = XGBClassifier()
    model.load_model(str(model_path))

    threshold = float(threshold_path.read_text().strip())
    features = json.loads(features_path.read_text())
    return model, threshold, features


def save_artifacts(model, threshold: float, feature_list):
    cfg = get_settings()
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(_p("model_file")))
    _p("threshold_file").write_text(str(float(threshold)))
    _p("feature_list_file").write_text(json.dumps(list(feature_list), indent=2))


def save_metadata(metadata: dict):
    """Write training metadata/metrics to artifacts/metadata.json"""
    cfg = get_settings()
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    meta_path = artifacts_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

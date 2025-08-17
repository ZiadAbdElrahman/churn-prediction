import pandas as pd
from datetime import datetime
from ...core.data_access import load_events
from ...core.features import build_features_for_user_date
from ...core.features import safe_div  # if needed elsewhere
from ..controller.types import PredictIn
from ...core.model_io import load_artifacts
from ...core.training_pipeline import train_from_events_json


class ChurnService:
    def __init__(self):
        try:
            self.model, self.threshold, self.feature_list = load_artifacts()
        except FileNotFoundError as e:
            self.model, self.threshold, self.feature_list = None, None, []
        self._events = load_events()  # cached by @lru_cache

    def predict_user_date(self, req: PredictIn):
        cutoff = pd.to_datetime(req.date).tz_localize(None)
        print(cutoff)

        # build one-row feature frame for this user/date
        feat_row = build_features_for_user_date(self._events, req.user_id, cutoff)
        # align columns to training order (missing -> 0)
        X = feat_row.reindex(columns=self.feature_list, fill_value=0)

        prob = float(self.model.predict_proba(X)[:, 1][0])
        label = int(prob >= self.threshold)

        return {
            "user_id": req.user_id,
            "date": cutoff.to_pydatetime(),
            "probability": prob,
            "label": label,
            "threshold": float(self.threshold),
        }

    def train_from_json(self):
        report = train_from_events_json()
        # hot-reload newly saved artifacts
        self.model, self.threshold, self.feature_list = load_artifacts()
        return report

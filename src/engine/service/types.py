# from pydantic import BaseModel
# from typing import List, Dict

# class PredictUserIn(BaseModel):
#     features: List[Dict]   # already engineered rows

# class PredictResponse(BaseModel):
#     probabilities: List[float]
#     predictions: List[int]
#     threshold: float

# class TrainReport(BaseModel):
#     val_pr_auc: float
#     threshold: float
#     scale_pos_weight: float
#     pos_rate_train: float

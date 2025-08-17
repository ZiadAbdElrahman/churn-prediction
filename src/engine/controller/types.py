from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class PredictIn(BaseModel):
    user_id: str = Field(..., description="User identifier")
    date: datetime = Field(
        ..., description="ISO date/time; features use data up to this date (inclusive)"
    )


class PredictOut(BaseModel):
    user_id: str
    date: datetime
    probability: float
    label: int
    threshold: float


class TrainOut(BaseModel):
    val_pr_auc: float
    threshold: float
    scale_pos_weight: float
    pos_rate_train: float

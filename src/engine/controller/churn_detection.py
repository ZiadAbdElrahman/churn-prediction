from fastapi import APIRouter, HTTPException
from .types import PredictIn, PredictOut, TrainOut
from ..service.churn_detection import ChurnService

router = APIRouter()
_service = None


def get_service():
    global _service
    if _service is None:
        _service = ChurnService()
    return _service


@router.post("/predict", response_model=PredictOut)
def predict(req: PredictIn):
    try:
        resp = get_service().predict_user_date(req)
        return PredictOut(**resp)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/train", response_model=TrainOut)
def train():
    try:
        rep = get_service().train_from_json()
        return TrainOut(
            val_pr_auc=rep["val_pr_auc"],
            threshold=rep["threshold"],
            scale_pos_weight=rep["scale_pos_weight"],
            pos_rate_train=rep["pos_rate_train"],
        )
    except Exception as e:
        print(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from smart_train_ai.schema import SensorRecord, PredictionResult
from smart_train_ai.inference.predictor import Predictor

app = FastAPI(
    title="Smart Train AI Fault Detection & Localization API",
    description="Predictive maintenance REST API for smart rolling stock physical prototype fault detection and localization.",
    version="1.0.0",
)

predictor: Predictor = None


def get_predictor() -> Predictor:
    """Lazy loader for Predictor singleton instance."""
    global predictor
    if predictor is None:
        try:
            predictor = Predictor(models_dir="models_artifacts")
        except Exception as e:
            print(f"Warning initializing predictor: {e}")
            predictor = None
    return predictor


@app.on_event("startup")
def startup_event():
    """Initializes ML Predictor pipeline on server startup."""
    get_predictor()


@app.get("/health")
def get_health() -> Dict[str, str]:
    """Returns API service operational health."""
    return {"status": "ok", "service": "Smart Train AI Inference API", "version": "1.0.0"}


@app.get("/model-info")
def get_model_info() -> Dict[str, Any]:
    """Returns loaded model versions, feature metadata, and pipeline capabilities."""
    pred = get_predictor()
    if pred is None:
        return {"status": "models_not_loaded", "message": "Models not yet trained or loaded."}

    return {
        "status": "ready",
        "anomaly_detector": "Isolation Forest" if pred.anomaly_detector else "None",
        "fault_classifier": "XGBoost Multiclass" if pred.fault_classifier else "None",
        "fault_localizer": "XGBoost Localizer" if pred.fault_localizer else "None",
        "sampling_hz": pred.sampling_hz,
        "supported_faults": [
            "NORMAL", "WHEEL_FLAT", "AXLE_MISALIGNMENT", "BEARING_FAULT",
            "BRAKE_ABNORMAL", "SUSPENSION_FAULT", "MOTOR_FAULT", "BODY_DAMAGE", "UNKNOWN"
        ],
    }


@app.post("/predict", response_model=PredictionResult)
def predict_telemetry(records: List[SensorRecord]) -> PredictionResult:
    """
    Accepts a sequence of telemetry SensorRecord items for a 5-second window
    and returns full fault classification, anomaly score, location, health index, and evidence.
    """
    pred = get_predictor()
    if pred is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor models not available or loaded.",
        )

    if not records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sensor records list cannot be empty.",
        )

    try:
        result = pred.predict_window(records)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}",
        )


@app.post("/batch-predict", response_model=List[PredictionResult])
def batch_predict_telemetry(batches: List[List[SensorRecord]]) -> List[PredictionResult]:
    """Accepts multiple 5-second window batches and returns predictions for each batch."""
    pred = get_predictor()
    if pred is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor models not available or loaded.",
        )

    results = []
    for batch in batches:
        res = pred.predict_window(batch)
        results.append(res)
    return results

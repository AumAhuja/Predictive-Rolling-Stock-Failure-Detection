from typing import Dict, Any, Optional
from pydantic import BaseModel
from smart_train_ai.schema import PredictionResult, FaultType, FaultLocation


class VisionPrediction(BaseModel):
    visual_fault: str  # e.g. BODY_DAMAGE, WHEEL_FLAT, NORMAL
    location: str  # e.g. BODY, FRONT_BOGIE_AXLE_1
    confidence: float  # 0.0 to 1.0
    image_url: Optional[str] = None


class FusionEngine:
    """Fuses multi-modal telemetry prediction evidence with computer vision predictions."""

    @staticmethod
    def fuse(
        sensor_pred: PredictionResult,
        vision_pred: Optional[VisionPrediction] = None,
    ) -> Dict[str, Any]:
        """Combines sensor diagnostic predictions with vision findings into a unified diagnosis."""
        final_diag = sensor_pred.model_dump()

        if vision_pred is None:
            final_diag["vision_evidence"] = "No visual inspection telemetry available"
            return final_diag

        # If vision confidence is high (> 0.85) and agrees with or refines sensor prediction
        sensor_fault = sensor_pred.fault_type.value if hasattr(sensor_pred.fault_type, "value") else str(sensor_pred.fault_type)
        vision_fault = vision_pred.visual_fault

        evidence = list(sensor_pred.evidence)
        evidence.append(f"Visual Inspection: Detected '{vision_fault}' at {vision_pred.location} ({vision_pred.confidence * 100:.0f}% confidence)")

        # Fuse confidence
        if sensor_fault == vision_fault:
            fused_conf = min(1.0, sensor_pred.confidence * 0.5 + vision_pred.confidence * 0.5 + 0.1)
            final_diag["confidence"] = round(fused_conf, 2)
            final_diag["fault_type"] = sensor_fault
        elif vision_pred.confidence > 0.85 and sensor_pred.confidence < 0.60:
            final_diag["fault_type"] = vision_fault
            final_diag["fault_location"] = vision_pred.location
            final_diag["confidence"] = round(vision_pred.confidence, 2)

        final_diag["evidence"] = evidence
        final_diag["vision_prediction"] = vision_pred.model_dump()

        return final_diag

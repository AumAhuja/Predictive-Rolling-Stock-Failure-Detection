import pytest
import os
import tempfile
import pandas as pd
from smart_train_ai.schema import FaultType, FaultLocation
from smart_train_ai.simulator.generator import generate_run_dataframe, create_simulator
from smart_train_ai.training.train_anomaly import train_anomaly_model
from smart_train_ai.training.train_classifier import train_classifier_model
from smart_train_ai.training.train_localization import train_localization_model
from smart_train_ai.inference.predictor import Predictor


def test_end_to_end_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = os.path.join(tmp_dir, "data")
        models_dir = os.path.join(tmp_dir, "models")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # 1. Generate Synthetic Dataset for normal and bearing fault (4 runs total for training split balance)
        for i in range(2):
            df_norm = generate_run_dataframe(FaultType.NORMAL, run_id=f"RUN_00{i+1}_NORMAL", duration_seconds=15.0, seed=10+i)
            df_norm.to_csv(os.path.join(data_dir, f"RUN_00{i+1}_NORMAL.csv"), index=False)

            df_bearing = generate_run_dataframe(
                FaultType.BEARING_FAULT,
                run_id=f"RUN_00{i+3}_BEARING_FAULT",
                duration_seconds=15.0,
                fault_severity=2,
                fault_location=FaultLocation.REAR_BOGIE_AXLE_2,
                seed=20+i,
            )
            df_bearing.to_csv(os.path.join(data_dir, f"RUN_00{i+3}_BEARING_FAULT.csv"), index=False)

        # 2. Train Models
        train_anomaly_model(data_dir=data_dir, model_output_path=os.path.join(models_dir, "anomaly_model.joblib"))
        train_classifier_model(data_dir=data_dir, model_output_path=os.path.join(models_dir, "classifier_model.joblib"))
        train_localization_model(data_dir=data_dir, model_output_path=os.path.join(models_dir, "localization_model.joblib"))

        # 3. Create Predictor and Execute Inference on a synthetic stream
        predictor = Predictor(models_dir=models_dir)

        sim = create_simulator(
            FaultType.BEARING_FAULT,
            run_id="STREAM_TEST",
            duration_seconds=5.0,
            fault_severity=2,
            fault_location=FaultLocation.REAR_BOGIE_AXLE_2,
            seed=99,
        )
        stream_records = sim.generate_records()

        # 4. Predict
        result = predictor.predict_window(stream_records)

        # 5. Verify Diagnostic JSON output payload
        assert result.train_id == "TRAIN001"
        assert isinstance(result.is_anomaly, bool)
        assert 0.0 <= result.anomaly_score <= 1.0
        assert result.fault_type in [FaultType.BEARING_FAULT, FaultType.UNKNOWN]
        assert 0 <= result.health_score <= 100
        assert result.status in ["HEALTHY", "WARNING", "CRITICAL"]
        assert result.maintenance_priority in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert len(result.evidence) > 0
        assert result.raw_window is not None
        assert "channels" in result.raw_window

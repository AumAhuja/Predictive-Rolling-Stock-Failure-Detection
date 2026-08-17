import pytest
from smart_train_ai.schema import (
    SensorRecord,
    FaultType,
    FaultLocation,
    PredictionResult,
)


def test_sensor_record_creation():
    rec = SensorRecord(
        timestamp=100.0,
        run_id="RUN_001",
        train_id="TRAIN001",
        imu_x=0.01,
        imu_y=-0.02,
        imu_z=1.01,
        motor_temperature=28.5,
        bearing_temperature=26.2,
        ambient_temperature=24.0,
        rpm=120.0,
        voltage=5.02,
        current=0.25,
        power=1.25,
        load=1.5,
        zone="ZONE_01",
        fault_type=FaultType.NORMAL,
        fault_location=FaultLocation.NONE,
        fault_severity=0,
    )
    assert rec.run_id == "RUN_001"
    assert rec.fault_type == FaultType.NORMAL
    assert rec.imu_z == 1.01


def test_prediction_result_schema():
    res = PredictionResult(
        train_id="TRAIN001",
        is_anomaly=True,
        anomaly_score=0.82,
        fault_type=FaultType.BEARING_FAULT,
        fault_location=FaultLocation.REAR_BOGIE_AXLE_2,
        confidence=0.87,
        health_score=81,
        status="WARNING",
        maintenance_priority="MEDIUM",
        evidence=["Vibration RMS increased", "Bearing temperature slope elevated"],
    )
    assert res.is_anomaly is True
    assert res.fault_type == FaultType.BEARING_FAULT
    assert res.fault_location == FaultLocation.REAR_BOGIE_AXLE_2

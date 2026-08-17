from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class FaultType(str, Enum):
    NORMAL = "NORMAL"
    WHEEL_FLAT = "WHEEL_FLAT"
    AXLE_MISALIGNMENT = "AXLE_MISALIGNMENT"
    BEARING_FAULT = "BEARING_FAULT"
    BRAKE_ABNORMAL = "BRAKE_ABNORMAL"
    SUSPENSION_FAULT = "SUSPENSION_FAULT"
    MOTOR_FAULT = "MOTOR_FAULT"
    BODY_DAMAGE = "BODY_DAMAGE"
    UNKNOWN = "UNKNOWN"


class FaultLocation(str, Enum):
    NONE = "NONE"
    MOTOR = "MOTOR"
    FRONT_BOGIE = "FRONT_BOGIE"
    FRONT_BOGIE_AXLE_1 = "FRONT_BOGIE_AXLE_1"
    FRONT_BOGIE_AXLE_2 = "FRONT_BOGIE_AXLE_2"
    REAR_BOGIE = "REAR_BOGIE"
    REAR_BOGIE_AXLE_1 = "REAR_BOGIE_AXLE_1"
    REAR_BOGIE_AXLE_2 = "REAR_BOGIE_AXLE_2"
    BODY = "BODY"
    BRAKE = "BRAKE"


class FaultSeverity(int, Enum):
    NORMAL = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3


class SensorRecord(BaseModel):
    timestamp: float = Field(..., description="UNIX timestamp in seconds")
    run_id: str = Field(..., description="Unique run identifier for telemetry session")
    train_id: str = Field(default="TRAIN001", description="Rolling stock physical asset ID")

    imu_x: float = Field(..., description="IMU Acceleration X (g or m/s^2)")
    imu_y: float = Field(..., description="IMU Acceleration Y (g or m/s^2)")
    imu_z: float = Field(..., description="IMU Acceleration Z (g or m/s^2)")

    motor_temperature: float = Field(..., description="DS18B20 Motor Temperature (°C)")
    bearing_temperature: float = Field(..., description="DS18B20 Bearing Temperature (°C)")
    ambient_temperature: float = Field(..., description="DHT11 Ambient Temperature (°C)")

    rpm: float = Field(..., description="Wheel/Motor speed derived from reed switch (RPM)")

    voltage: float = Field(..., description="INA219 Supply Voltage (V)")
    current: float = Field(..., description="INA219 Draw Current (A)")
    power: float = Field(..., description="INA219 Electrical Power (W)")

    load: float = Field(..., description="HX711 Load cell measurement (kg)")
    zone: str = Field(default="ZONE_01", description="Track inspection zone or IR trigger marker")

    fault_type: FaultType = Field(default=FaultType.NORMAL, description="Ground truth fault label")
    fault_location: FaultLocation = Field(default=FaultLocation.NONE, description="Ground truth fault location")
    fault_severity: int = Field(default=0, description="Ground truth fault severity (0-3)")


class SensorWindow(BaseModel):
    run_id: str
    train_id: str
    start_timestamp: float
    end_timestamp: float
    records: List[SensorRecord]


class EvidenceItem(BaseModel):
    feature: str
    description: str
    importance: Optional[float] = None
    observed_value: Optional[float] = None


class PredictionResult(BaseModel):
    train_id: str
    is_anomaly: bool
    anomaly_score: float
    fault_type: FaultType
    fault_location: FaultLocation
    confidence: float
    health_score: int
    status: str  # HEALTHY, WARNING, CRITICAL
    maintenance_priority: str  # LOW, MEDIUM, HIGH, CRITICAL
    evidence: List[str]
    raw_window: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Preserved raw temporal time-series matrix for future 1D CNN / TCN / LSTM integration"
    )

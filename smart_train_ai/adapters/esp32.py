import pandas as pd
from typing import Dict, Any, List
from smart_train_ai.adapters.base import DataSource
from smart_train_ai.schema import SensorRecord, FaultType, FaultLocation


class ESP32DataSource(DataSource):
    """Data source adapter parsing live ESP32 microcontroller telemetry payloads."""

    def __init__(self):
        self._buffer: List[SensorRecord] = []

    def ingest_payload(self, payload: Dict[str, Any]) -> SensorRecord:
        """Parses ESP32 raw JSON payload into SensorRecord and appends to local buffer."""
        record = SensorRecord(
            timestamp=float(payload.get("timestamp", 0.0)),
            run_id=str(payload.get("run_id", "ESP32_RUN")),
            train_id=str(payload.get("train_id", "TRAIN001")),
            imu_x=float(payload.get("imu_x", 0.0)),
            imu_y=float(payload.get("imu_y", 0.0)),
            imu_z=float(payload.get("imu_z", 1.0)),
            motor_temperature=float(payload.get("motor_temp", payload.get("motor_temperature", 25.0))),
            bearing_temperature=float(payload.get("bearing_temp", payload.get("bearing_temperature", 25.0))),
            ambient_temperature=float(payload.get("ambient_temp", payload.get("ambient_temperature", 25.0))),
            rpm=float(payload.get("rpm", 0.0)),
            voltage=float(payload.get("voltage", 5.0)),
            current=float(payload.get("current", 0.2)),
            power=float(payload.get("power", 1.0)),
            load=float(payload.get("load", 1.0)),
            zone=str(payload.get("zone", "ZONE_01")),
            fault_type=FaultType(payload.get("fault_type", "UNKNOWN")),
            fault_location=FaultLocation(payload.get("fault_location", "NONE")),
            fault_severity=int(payload.get("fault_severity", 0)),
        )
        self._buffer.append(record)
        return record

    def fetch_latest_record(self) -> SensorRecord:
        if not self._buffer:
            raise ValueError("No ESP32 sensor records ingested yet.")
        return self._buffer[-1]

    def fetch_window_df(self, duration_seconds: float = 5.0) -> pd.DataFrame:
        if not self._buffer:
            return pd.DataFrame()
        data = [r.model_dump() for r in self._buffer]
        df = pd.DataFrame(data)
        if df.empty:
            return df
        max_t = df["timestamp"].max()
        window_df = df[df["timestamp"] >= (max_t - duration_seconds)]
        return window_df

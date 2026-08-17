import pandas as pd
from typing import Optional
from smart_train_ai.adapters.base import DataSource
from smart_train_ai.schema import SensorRecord, FaultType, FaultLocation
from smart_train_ai.simulator.generator import generate_run_dataframe, create_simulator


class SyntheticDataSource(DataSource):
    """Data source adapter reading from synthetic physics simulators."""

    def __init__(
        self,
        fault_type: FaultType = FaultType.NORMAL,
        run_id: str = "SYNTH_RUN_001",
        speed_rpm: float = 120.0,
        load_kg: float = 1.5,
        fault_severity: int = 0,
        fault_location: Optional[FaultLocation] = None,
    ):
        self.fault_type = fault_type
        self.run_id = run_id
        self.speed_rpm = speed_rpm
        self.load_kg = load_kg
        self.fault_severity = fault_severity
        self.fault_location = fault_location

    def fetch_latest_record(self) -> SensorRecord:
        sim = create_simulator(
            fault_type=self.fault_type,
            run_id=self.run_id,
            duration_seconds=1.0,
            speed_rpm=self.speed_rpm,
            load_kg=self.load_kg,
            fault_severity=self.fault_severity,
            fault_location=self.fault_location,
        )
        records = sim.generate_records()
        return records[-1]

    def fetch_window_df(self, duration_seconds: float = 5.0) -> pd.DataFrame:
        return generate_run_dataframe(
            fault_type=self.fault_type,
            run_id=self.run_id,
            duration_seconds=duration_seconds,
            speed_rpm=self.speed_rpm,
            load_kg=self.load_kg,
            fault_severity=self.fault_severity,
            fault_location=self.fault_location,
        )

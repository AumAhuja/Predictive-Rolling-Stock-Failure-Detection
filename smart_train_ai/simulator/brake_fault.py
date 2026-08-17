import numpy as np
from typing import List
from smart_train_ai.schema import SensorRecord, FaultType, FaultLocation
from smart_train_ai.simulator.base import BaseSimulator


class BrakeAbnormalSimulator(BaseSimulator):
    """Brake fault simulator modelling mechanical resistance drag (Current UP, RPM DOWN, Temp UP)."""

    def __init__(self, *args, fault_location: FaultLocation = FaultLocation.BRAKE, **kwargs):
        super().__init__(*args, **kwargs)
        self._location = fault_location

    @property
    def fault_type(self) -> FaultType:
        return FaultType.BRAKE_ABNORMAL

    @property
    def default_location(self) -> FaultLocation:
        return self._location

    def generate_records(self) -> List[SensorRecord]:
        num_steps = int(self.duration_seconds * self.imu_hz)
        timestamps = np.linspace(0.0, self.duration_seconds, num_steps, endpoint=False)

        severity_scale = float(max(self.fault_severity, 1))

        # Drag reduces speed and increases motor load resistance
        drag_factor = 1.0 - 0.08 * severity_scale
        rpm = (self.speed_rpm * drag_factor) + self.rng.normal(0.0, 1.5 * severity_scale, num_steps)
        rpm = np.maximum(rpm, 0.0)

        # High current draw due to mechanical strain
        base_current = 0.2 + (self.load_kg * 0.05) + 0.15 * severity_scale
        current = base_current + self.rng.normal(0.0, 0.03 * severity_scale, num_steps)
        voltage = 5.0 - (0.05 * severity_scale) + self.rng.normal(0.0, 0.02, num_steps)
        power = voltage * current

        # High thermal rise in motor and brake/bearing area
        motor_temp = self.ambient_temp + (12.0 + 6.0 * severity_scale) * (1.0 - np.exp(-timestamps / 12.0)) + self.rng.normal(0.0, 0.1, num_steps)
        bearing_temp = self.ambient_temp + (10.0 + 5.0 * severity_scale) * (1.0 - np.exp(-timestamps / 12.0)) + self.rng.normal(0.0, 0.1, num_steps)
        ambient_temp = self.ambient_temp + self.rng.normal(0.0, 0.05, num_steps)

        # Brake chatter in IMU X/Z
        imu_x = 0.2 * severity_scale * np.sin(2 * np.pi * 15.0 * timestamps) + self.rng.normal(0.0, 0.1, num_steps)
        imu_y = self.rng.normal(0.0, 0.08, num_steps)
        imu_z = 1.0 + 0.15 * severity_scale * np.cos(2 * np.pi * 15.0 * timestamps) + self.rng.normal(0.0, 0.1, num_steps)

        load = self.load_kg + 0.1 * severity_scale + self.rng.normal(0.0, 0.03, num_steps)

        records = []
        for i in range(num_steps):
            records.append(
                SensorRecord(
                    timestamp=float(timestamps[i]),
                    run_id=self.run_id,
                    train_id=self.train_id,
                    imu_x=float(imu_x[i]),
                    imu_y=float(imu_y[i]),
                    imu_z=float(imu_z[i]),
                    motor_temperature=float(motor_temp[i]),
                    bearing_temperature=float(bearing_temp[i]),
                    ambient_temperature=float(ambient_temp[i]),
                    rpm=float(rpm[i]),
                    voltage=float(voltage[i]),
                    current=float(current[i]),
                    power=float(power[i]),
                    load=float(load[i]),
                    zone="ZONE_02",
                    fault_type=self.fault_type,
                    fault_location=self.default_location,
                    fault_severity=self.fault_severity,
                )
            )
        return records

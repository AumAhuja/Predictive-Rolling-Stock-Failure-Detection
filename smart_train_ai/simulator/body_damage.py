import numpy as np
from typing import List
from smart_train_ai.schema import SensorRecord, FaultType, FaultLocation
from smart_train_ai.simulator.base import BaseSimulator


class BodyDamageSimulator(BaseSimulator):
    """Body damage simulator modelling structural chassis resonance and aerodynamic dynamic tilt."""

    def __init__(self, *args, fault_location: FaultLocation = FaultLocation.BODY, **kwargs):
        super().__init__(*args, **kwargs)
        self._location = fault_location

    @property
    def fault_type(self) -> FaultType:
        return FaultType.BODY_DAMAGE

    @property
    def default_location(self) -> FaultLocation:
        return self._location

    def generate_records(self) -> List[SensorRecord]:
        num_steps = int(self.duration_seconds * self.imu_hz)
        timestamps = np.linspace(0.0, self.duration_seconds, num_steps, endpoint=False)

        severity_scale = float(max(self.fault_severity, 1))

        # Structural flex resonance (~8 Hz)
        flex_res = 0.35 * severity_scale * np.sin(2 * np.pi * 8.0 * timestamps)

        # Permanent offset tilt in X/Y axes
        imu_x = 0.15 * severity_scale + flex_res + self.rng.normal(0.0, 0.1, num_steps)
        imu_y = -0.10 * severity_scale + 0.5 * flex_res + self.rng.normal(0.0, 0.1, num_steps)
        imu_z = 1.0 + 0.2 * severity_scale * np.cos(2 * np.pi * 8.0 * timestamps) + self.rng.normal(0.0, 0.1, num_steps)

        rpm = self.speed_rpm + self.rng.normal(0.0, 0.6, num_steps)
        motor_temp = self.ambient_temp + 8.0 * (1.0 - np.exp(-timestamps / 15.0)) + self.rng.normal(0.0, 0.1, num_steps)
        bearing_temp = self.ambient_temp + 4.5 * (1.0 - np.exp(-timestamps / 18.0)) + self.rng.normal(0.0, 0.1, num_steps)
        ambient_temp = self.ambient_temp + self.rng.normal(0.0, 0.05, num_steps)

        voltage = 5.0 + self.rng.normal(0.0, 0.02, num_steps)
        current = 0.22 + self.rng.normal(0.0, 0.015, num_steps)
        power = voltage * current

        load = self.load_kg + 0.05 * severity_scale + self.rng.normal(0.0, 0.05, num_steps)

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
                    zone="ZONE_01",
                    fault_type=self.fault_type,
                    fault_location=self.default_location,
                    fault_severity=self.fault_severity,
                )
            )
        return records

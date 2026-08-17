import numpy as np
from typing import List
from smart_train_ai.schema import SensorRecord, FaultType, FaultLocation
from smart_train_ai.simulator.base import BaseSimulator


class SuspensionFaultSimulator(BaseSimulator):
    """Suspension fault simulator producing vertical/lateral bounce shocks under load."""

    def __init__(self, *args, fault_location: FaultLocation = FaultLocation.REAR_BOGIE, **kwargs):
        super().__init__(*args, **kwargs)
        self._location = fault_location

    @property
    def fault_type(self) -> FaultType:
        return FaultType.SUSPENSION_FAULT

    @property
    def default_location(self) -> FaultLocation:
        return self._location

    def generate_records(self) -> List[SensorRecord]:
        num_steps = int(self.duration_seconds * self.imu_hz)
        timestamps = np.linspace(0.0, self.duration_seconds, num_steps, endpoint=False)

        severity_scale = float(max(self.fault_severity, 1))

        # Low frequency resonance oscillation (2-5 Hz) + random transient shock spikes
        bounce_freq = 3.5
        resonance = 0.5 * severity_scale * np.sin(2 * np.pi * bounce_freq * timestamps)

        # Random shock bumps
        num_bumps = int(self.duration_seconds * 1.5)
        bump_indices = self.rng.choice(num_steps, size=num_bumps, replace=False)
        shocks = np.zeros(num_steps)
        shocks[bump_indices] = self.rng.uniform(0.8, 1.8, size=num_bumps) * severity_scale

        imu_x = self.rng.normal(0.0, 0.1, num_steps)
        imu_y = resonance * 0.7 + self.rng.normal(0.0, 0.15 * severity_scale, num_steps)
        imu_z = 1.0 + resonance + shocks + self.rng.normal(0.0, 0.2 * severity_scale, num_steps)

        rpm = self.speed_rpm + self.rng.normal(0.0, 0.8, num_steps)
        motor_temp = self.ambient_temp + 8.5 * (1.0 - np.exp(-timestamps / 15.0)) + self.rng.normal(0.0, 0.1, num_steps)
        bearing_temp = self.ambient_temp + 5.0 * (1.0 - np.exp(-timestamps / 18.0)) + self.rng.normal(0.0, 0.1, num_steps)
        ambient_temp = self.ambient_temp + self.rng.normal(0.0, 0.05, num_steps)

        voltage = 5.0 + self.rng.normal(0.0, 0.02, num_steps)
        current = 0.23 + self.rng.normal(0.0, 0.02, num_steps)
        power = voltage * current

        load = self.load_kg + 0.2 * severity_scale * np.sin(2 * np.pi * bounce_freq * timestamps) + self.rng.normal(0.0, 0.1 * severity_scale, num_steps)

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

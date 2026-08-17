import numpy as np
from typing import List
from smart_train_ai.schema import SensorRecord, FaultType, FaultLocation
from smart_train_ai.simulator.base import BaseSimulator


class WheelFlatSimulator(BaseSimulator):
    """Wheel flat fault simulator introducing periodic impulse shocks correlated with wheel rotation."""

    def __init__(self, *args, fault_location: FaultLocation = FaultLocation.FRONT_BOGIE_AXLE_1, **kwargs):
        super().__init__(*args, **kwargs)
        self._location = fault_location

    @property
    def fault_type(self) -> FaultType:
        return FaultType.WHEEL_FLAT

    @property
    def default_location(self) -> FaultLocation:
        return self._location

    def generate_records(self) -> List[SensorRecord]:
        num_steps = int(self.duration_seconds * self.imu_hz)
        timestamps = np.linspace(0.0, self.duration_seconds, num_steps, endpoint=False)

        # Rotation frequency (Hz)
        f_wheel = max(self.speed_rpm / 60.0, 0.5)
        period = 1.0 / f_wheel

        # Periodic impact pulses
        impulse_phase = (timestamps % period) / period
        # Pulse peak occurs around phase 0
        pulse = np.exp(-((impulse_phase - 0.05) ** 2) / (2 * (0.01 ** 2)))

        severity_scale = float(max(self.fault_severity, 1))

        imu_x = self.rng.normal(0.0, 0.06, num_steps) + 0.3 * severity_scale * pulse * self.rng.choice([-1, 1], num_steps)
        imu_y = self.rng.normal(0.0, 0.06, num_steps) + 0.2 * severity_scale * pulse
        imu_z = 1.0 + self.rng.normal(0.0, 0.1, num_steps) + 0.8 * severity_scale * pulse

        rpm = self.speed_rpm + self.rng.normal(0.0, 1.2 * severity_scale, num_steps)
        motor_temp = self.ambient_temp + 9.0 * (1.0 - np.exp(-timestamps / 15.0)) + self.rng.normal(0.0, 0.1, num_steps)
        bearing_temp = self.ambient_temp + (5.0 + 2.0 * severity_scale) * (1.0 - np.exp(-timestamps / 18.0)) + self.rng.normal(0.0, 0.1, num_steps)
        ambient_temp = self.ambient_temp + self.rng.normal(0.0, 0.05, num_steps)

        voltage = 5.0 + self.rng.normal(0.0, 0.03, num_steps)
        base_current = 0.22 + (self.load_kg * 0.05) + (severity_scale * 0.03)
        current = base_current + self.rng.normal(0.0, 0.02, num_steps)
        power = voltage * current

        load = self.load_kg + 0.15 * severity_scale * pulse + self.rng.normal(0.0, 0.03, num_steps)

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
                    zone="ZONE_01" if timestamps[i] < self.duration_seconds / 2 else "ZONE_02",
                    fault_type=self.fault_type,
                    fault_location=self.default_location,
                    fault_severity=self.fault_severity,
                )
            )
        return records

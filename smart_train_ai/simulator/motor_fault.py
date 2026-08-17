import numpy as np
from typing import List
from smart_train_ai.schema import SensorRecord, FaultType, FaultLocation
from smart_train_ai.simulator.base import BaseSimulator


class MotorFaultSimulator(BaseSimulator):
    """Motor fault simulator introducing electromagnetic ripple, thermal rise, and speed instability."""

    def __init__(self, *args, fault_location: FaultLocation = FaultLocation.MOTOR, **kwargs):
        super().__init__(*args, **kwargs)
        self._location = fault_location

    @property
    def fault_type(self) -> FaultType:
        return FaultType.MOTOR_FAULT

    @property
    def default_location(self) -> FaultLocation:
        return self._location

    def generate_records(self) -> List[SensorRecord]:
        num_steps = int(self.duration_seconds * self.imu_hz)
        timestamps = np.linspace(0.0, self.duration_seconds, num_steps, endpoint=False)

        severity_scale = float(max(self.fault_severity, 1))

        # Motor speed hunting / instability
        rpm_instability = 5.0 * severity_scale * np.sin(2 * np.pi * 0.8 * timestamps) + self.rng.normal(0.0, 2.0 * severity_scale, num_steps)
        rpm = self.speed_rpm + rpm_instability
        rpm = np.maximum(rpm, 0.0)

        # High motor temperature rise
        motor_temp = self.ambient_temp + (15.0 + 5.0 * severity_scale) * (1.0 - np.exp(-timestamps / 8.0)) + self.rng.normal(0.0, 0.15, num_steps)
        bearing_temp = self.ambient_temp + 6.0 * (1.0 - np.exp(-timestamps / 15.0)) + self.rng.normal(0.0, 0.1, num_steps)
        ambient_temp = self.ambient_temp + self.rng.normal(0.0, 0.05, num_steps)

        # Electrical current ripple at motor commutation frequency (~50 Hz)
        current_ripple = 0.08 * severity_scale * np.sin(2 * np.pi * 50.0 * timestamps)
        base_current = 0.25 + 0.05 * severity_scale
        current = base_current + current_ripple + self.rng.normal(0.0, 0.03 * severity_scale, num_steps)
        current = np.maximum(current, 0.01)

        voltage = 5.0 + self.rng.normal(0.0, 0.04 * severity_scale, num_steps)
        power = voltage * current

        # Motor housing vibration
        motor_vibe = 0.3 * severity_scale * np.sin(2 * np.pi * 50.0 * timestamps)
        imu_x = motor_vibe + self.rng.normal(0.0, 0.1, num_steps)
        imu_y = 0.5 * motor_vibe + self.rng.normal(0.0, 0.1, num_steps)
        imu_z = 1.0 + motor_vibe + self.rng.normal(0.0, 0.12, num_steps)

        load = self.load_kg + self.rng.normal(0.0, 0.03, num_steps)

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

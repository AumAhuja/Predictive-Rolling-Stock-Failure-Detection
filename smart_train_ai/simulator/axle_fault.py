import numpy as np
from typing import List
from smart_train_ai.schema import SensorRecord, FaultType, FaultLocation
from smart_train_ai.simulator.base import BaseSimulator


class AxleMisalignmentSimulator(BaseSimulator):
    """Axle misalignment simulator introducing directional harmonic vibration and load modulation."""

    def __init__(self, *args, fault_location: FaultLocation = FaultLocation.FRONT_BOGIE_AXLE_2, **kwargs):
        super().__init__(*args, **kwargs)
        self._location = fault_location

    @property
    def fault_type(self) -> FaultType:
        return FaultType.AXLE_MISALIGNMENT

    @property
    def default_location(self) -> FaultLocation:
        return self._location

    def generate_records(self) -> List[SensorRecord]:
        num_steps = int(self.duration_seconds * self.imu_hz)
        timestamps = np.linspace(0.0, self.duration_seconds, num_steps, endpoint=False)

        f_rot = max(self.speed_rpm / 60.0, 0.5)
        severity_scale = float(max(self.fault_severity, 1))

        # Rotational wobble harmonics
        wobble_x = 0.4 * severity_scale * np.sin(2 * np.pi * f_rot * timestamps)
        wobble_y = 0.3 * severity_scale * np.cos(2 * np.pi * f_rot * timestamps)
        harmonic_2x = 0.15 * severity_scale * np.sin(4 * np.pi * f_rot * timestamps)

        imu_x = wobble_x + harmonic_2x + self.rng.normal(0.0, 0.08, num_steps)
        imu_y = wobble_y + self.rng.normal(0.0, 0.08, num_steps)
        imu_z = 1.0 + 0.1 * severity_scale * np.sin(2 * np.pi * f_rot * timestamps) + self.rng.normal(0.0, 0.08, num_steps)

        rpm = self.speed_rpm + 1.5 * severity_scale * np.sin(2 * np.pi * f_rot * timestamps) + self.rng.normal(0.0, 0.5, num_steps)
        motor_temp = self.ambient_temp + 10.0 * (1.0 - np.exp(-timestamps / 15.0)) + self.rng.normal(0.0, 0.1, num_steps)
        bearing_temp = self.ambient_temp + (6.0 + 3.0 * severity_scale) * (1.0 - np.exp(-timestamps / 15.0)) + self.rng.normal(0.0, 0.1, num_steps)
        ambient_temp = self.ambient_temp + self.rng.normal(0.0, 0.05, num_steps)

        voltage = 5.0 + self.rng.normal(0.0, 0.02, num_steps)
        current = 0.25 + 0.04 * severity_scale + self.rng.normal(0.0, 0.02, num_steps)
        power = voltage * current

        load = self.load_kg + 0.1 * severity_scale * np.sin(2 * np.pi * f_rot * timestamps) + self.rng.normal(0.0, 0.04, num_steps)

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

import numpy as np
from typing import List
from smart_train_ai.schema import SensorRecord, FaultType, FaultLocation
from smart_train_ai.simulator.base import BaseSimulator


class NormalSimulator(BaseSimulator):
    """Generates physical telemetry for a healthy train operating under normal conditions."""

    @property
    def fault_type(self) -> FaultType:
        return FaultType.NORMAL

    @property
    def default_location(self) -> FaultLocation:
        return FaultLocation.NONE

    def generate_records(self) -> List[SensorRecord]:
        num_steps = int(self.duration_seconds * self.imu_hz)
        timestamps = np.linspace(0.0, self.duration_seconds, num_steps, endpoint=False)

        # Baseline IMU vibration (small sensor noise, gravity on Z)
        imu_x = self.rng.normal(0.0, 0.03 * self.noise_level, num_steps)
        imu_y = self.rng.normal(0.0, 0.03 * self.noise_level, num_steps)
        imu_z = 1.0 + self.rng.normal(0.0, 0.04 * self.noise_level, num_steps)

        # Speed RPM: stable with minor speed control fluctuations
        rpm = self.speed_rpm + self.rng.normal(0.0, 0.2 * self.noise_level, num_steps)
        rpm = np.maximum(rpm, 0.0)

        # Temperatures: gradual physical warmup over long time constants
        motor_temp = self.ambient_temp + 8.0 * (1.0 - np.exp(-timestamps / 150.0)) + self.rng.normal(0.0, 0.05, num_steps)
        bearing_temp = self.ambient_temp + 4.0 * (1.0 - np.exp(-timestamps / 200.0)) + self.rng.normal(0.0, 0.05, num_steps)
        ambient_temp = self.ambient_temp + self.rng.normal(0.0, 0.02, num_steps)

        # Electrical: INA219 (Voltage ~5.0V, Current proportional to load & RPM)
        voltage = 5.0 + self.rng.normal(0.0, 0.01 * self.noise_level, num_steps)
        base_current = 0.2 + (self.load_kg * 0.05) + (rpm / 1000.0) * 0.1
        current = base_current + self.rng.normal(0.0, 0.005 * self.noise_level, num_steps)
        current = np.maximum(current, 0.01)
        power = voltage * current

        # Load: HX711 measurement with minor track bounce
        load = self.load_kg + self.rng.normal(0.0, 0.01 * self.noise_level, num_steps)

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
                    fault_severity=0,
                )
            )
        return records

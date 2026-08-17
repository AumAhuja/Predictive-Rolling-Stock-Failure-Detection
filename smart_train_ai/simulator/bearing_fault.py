import numpy as np
from typing import List
from smart_train_ai.schema import SensorRecord, FaultType, FaultLocation
from smart_train_ai.simulator.base import BaseSimulator


class BearingFaultSimulator(BaseSimulator):
    """Bearing fault simulator introducing high-frequency vibration chatter and significant thermal dissipation."""

    def __init__(self, *args, fault_location: FaultLocation = FaultLocation.REAR_BOGIE_AXLE_2, **kwargs):
        super().__init__(*args, **kwargs)
        self._location = fault_location

    @property
    def fault_type(self) -> FaultType:
        return FaultType.BEARING_FAULT

    @property
    def default_location(self) -> FaultLocation:
        return self._location

    def generate_records(self) -> List[SensorRecord]:
        num_steps = int(self.duration_seconds * self.imu_hz)
        timestamps = np.linspace(0.0, self.duration_seconds, num_steps, endpoint=False)

        severity_scale = float(max(self.fault_severity, 1))

        # High-frequency bearing chatter (30-45 Hz range)
        chatter_freq1 = 35.0
        chatter_freq2 = 42.0
        high_freq_vibe = (
            0.25 * severity_scale * np.sin(2 * np.pi * chatter_freq1 * timestamps) +
            0.20 * severity_scale * np.cos(2 * np.pi * chatter_freq2 * timestamps)
        )

        imu_x = high_freq_vibe + self.rng.normal(0.0, 0.15 * severity_scale, num_steps)
        imu_y = high_freq_vibe + self.rng.normal(0.0, 0.12 * severity_scale, num_steps)
        imu_z = 1.0 + high_freq_vibe + self.rng.normal(0.0, 0.18 * severity_scale, num_steps)

        rpm = self.speed_rpm + self.rng.normal(0.0, 1.0 * severity_scale, num_steps)

        # Bearing temperature rises rapidly with steep exponential curve
        bearing_rise = 18.0 * severity_scale * (1.0 - np.exp(-timestamps / 10.0))
        bearing_temp = self.ambient_temp + bearing_rise + self.rng.normal(0.0, 0.15, num_steps)
        motor_temp = self.ambient_temp + 11.0 * (1.0 - np.exp(-timestamps / 15.0)) + self.rng.normal(0.0, 0.1, num_steps)
        ambient_temp = self.ambient_temp + self.rng.normal(0.0, 0.05, num_steps)

        voltage = 5.0 + self.rng.normal(0.0, 0.03, num_steps)
        current = 0.28 + 0.05 * severity_scale + self.rng.normal(0.0, 0.02, num_steps)
        power = voltage * current

        load = self.load_kg + self.rng.normal(0.0, 0.04, num_steps)

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

import numpy as np
from abc import ABC, abstractmethod
from typing import List
from smart_train_ai.schema import SensorRecord, FaultType, FaultLocation


class BaseSimulator(ABC):
    """Abstract base class for physical train telemetry simulators."""

    def __init__(
        self,
        run_id: str,
        train_id: str = "TRAIN001",
        duration_seconds: float = 30.0,
        speed_rpm: float = 120.0,
        load_kg: float = 1.5,
        fault_severity: int = 0,
        ambient_temp: float = 25.0,
        noise_level: float = 1.0,
        imu_hz: float = 100.0,
        temp_hz: float = 1.0,
        electrical_hz: float = 10.0,
        load_hz: float = 80.0,
        seed: int = 42,
    ):
        self.run_id = run_id
        self.train_id = train_id
        self.duration_seconds = duration_seconds
        self.speed_rpm = speed_rpm
        self.load_kg = load_kg
        self.fault_severity = fault_severity
        self.ambient_temp = ambient_temp
        self.noise_level = noise_level
        self.imu_hz = imu_hz
        self.temp_hz = temp_hz
        self.electrical_hz = electrical_hz
        self.load_hz = load_hz
        self.rng = np.random.default_rng(seed)

    @property
    @abstractmethod
    def fault_type(self) -> FaultType:
        """Returns the primary fault class for this simulator."""
        pass

    @property
    @abstractmethod
    def default_location(self) -> FaultLocation:
        """Returns the default physical fault location."""
        pass

    @abstractmethod
    def generate_records(self) -> List[SensorRecord]:
        """Generates time-ordered list of SensorRecord objects for the run duration."""
        pass

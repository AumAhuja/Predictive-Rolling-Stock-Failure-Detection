import pandas as pd
from typing import List, Optional
from smart_train_ai.schema import FaultType, FaultLocation, SensorRecord
from smart_train_ai.simulator.normal import NormalSimulator
from smart_train_ai.simulator.wheel_fault import WheelFlatSimulator
from smart_train_ai.simulator.axle_fault import AxleMisalignmentSimulator
from smart_train_ai.simulator.bearing_fault import BearingFaultSimulator
from smart_train_ai.simulator.brake_fault import BrakeAbnormalSimulator
from smart_train_ai.simulator.suspension_fault import SuspensionFaultSimulator
from smart_train_ai.simulator.motor_fault import MotorFaultSimulator
from smart_train_ai.simulator.body_damage import BodyDamageSimulator

SIMULATOR_MAP = {
    FaultType.NORMAL: NormalSimulator,
    FaultType.WHEEL_FLAT: WheelFlatSimulator,
    FaultType.AXLE_MISALIGNMENT: AxleMisalignmentSimulator,
    FaultType.BEARING_FAULT: BearingFaultSimulator,
    FaultType.BRAKE_ABNORMAL: BrakeAbnormalSimulator,
    FaultType.SUSPENSION_FAULT: SuspensionFaultSimulator,
    FaultType.MOTOR_FAULT: MotorFaultSimulator,
    FaultType.BODY_DAMAGE: BodyDamageSimulator,
}


def create_simulator(
    fault_type: FaultType,
    run_id: str,
    train_id: str = "TRAIN001",
    duration_seconds: float = 30.0,
    speed_rpm: float = 120.0,
    load_kg: float = 1.5,
    fault_severity: int = 0,
    ambient_temp: float = 25.0,
    noise_level: float = 1.0,
    fault_location: Optional[FaultLocation] = None,
    seed: int = 42,
):
    """Instantiates a simulator instance for a given fault type."""
    sim_cls = SIMULATOR_MAP.get(fault_type, NormalSimulator)
    kwargs = dict(
        run_id=run_id,
        train_id=train_id,
        duration_seconds=duration_seconds,
        speed_rpm=speed_rpm,
        load_kg=load_kg,
        fault_severity=fault_severity,
        ambient_temp=ambient_temp,
        noise_level=noise_level,
        seed=seed,
    )
    if fault_location is not None and fault_type != FaultType.NORMAL:
        kwargs["fault_location"] = fault_location

    return sim_cls(**kwargs)


def generate_run_dataframe(
    fault_type: FaultType,
    run_id: str,
    duration_seconds: float = 30.0,
    speed_rpm: float = 120.0,
    load_kg: float = 1.5,
    fault_severity: int = 0,
    ambient_temp: float = 25.0,
    noise_level: float = 1.0,
    fault_location: Optional[FaultLocation] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates telemetry records for a single run and converts to a pandas DataFrame."""
    sim = create_simulator(
        fault_type=fault_type,
        run_id=run_id,
        duration_seconds=duration_seconds,
        speed_rpm=speed_rpm,
        load_kg=load_kg,
        fault_severity=fault_severity,
        ambient_temp=ambient_temp,
        noise_level=noise_level,
        fault_location=fault_location,
        seed=seed,
    )
    records: List[SensorRecord] = sim.generate_records()
    data = [r.model_dump() for r in records]
    df = pd.DataFrame(data)
    return df

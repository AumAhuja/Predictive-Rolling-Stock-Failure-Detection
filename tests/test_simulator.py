import pytest
from smart_train_ai.schema import FaultType, FaultLocation
from smart_train_ai.simulator.normal import NormalSimulator
from smart_train_ai.simulator.bearing_fault import BearingFaultSimulator
from smart_train_ai.simulator.generator import generate_run_dataframe


def test_normal_simulator():
    sim = NormalSimulator(run_id="RUN_NORM", duration_seconds=5.0, speed_rpm=120.0)
    records = sim.generate_records()
    assert len(records) == 500  # 5s @ 100Hz
    assert records[0].fault_type == FaultType.NORMAL
    assert records[0].fault_location == FaultLocation.NONE


def test_bearing_fault_simulator():
    sim = BearingFaultSimulator(
        run_id="RUN_BEARING",
        duration_seconds=5.0,
        fault_severity=2,
        fault_location=FaultLocation.REAR_BOGIE_AXLE_2,
    )
    records = sim.generate_records()
    assert len(records) == 500
    assert records[0].fault_type == FaultType.BEARING_FAULT
    assert records[0].fault_location == FaultLocation.REAR_BOGIE_AXLE_2
    # Check that bearing temperature rises over time
    assert records[-1].bearing_temperature > records[0].bearing_temperature


def test_generator_dataframe():
    df = generate_run_dataframe(
        fault_type=FaultType.WHEEL_FLAT,
        run_id="RUN_WHEEL",
        duration_seconds=5.0,
        fault_severity=1,
    )
    assert not df.empty
    assert "imu_z" in df.columns
    assert df["fault_type"].iloc[0] == FaultType.WHEEL_FLAT.value

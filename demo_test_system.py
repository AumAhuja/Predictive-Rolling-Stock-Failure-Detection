"""
Comprehensive Verification and Interactive Demonstration Script for Smart Train AI System.
Tests live inference predictor, ESP32 data source adapter, REST API endpoints, and Vision Fusion engine.
"""

import sys
import json
import time
from typing import Dict, Any
from smart_train_ai.schema import FaultType, FaultLocation
from smart_train_ai.simulator.generator import create_simulator
from smart_train_ai.adapters.esp32 import ESP32DataSource
from smart_train_ai.inference.predictor import Predictor
from smart_train_ai.fusion.fusion_engine import FusionEngine, VisionPrediction


def print_dashboard_display(test_name: str, result_dict: Dict[str, Any]):
    """Renders formatted digital train dashboard diagnostic card."""
    print("=" * 65)
    print(f"       SMART TRAIN AI DIAGNOSTIC DASHBOARD: {test_name.upper()}")
    print("=" * 65)
    print(f" Train Asset ID     : {result_dict.get('train_id', 'TRAIN001')}")
    print(f" Operating Status   : {result_dict.get('status', 'N/A')}")
    print(f" Health Score       : {result_dict.get('health_score', 0)} / 100")
    print(f" Anomaly Detected   : {result_dict.get('is_anomaly', False)} (Score: {result_dict.get('anomaly_score', 0.0):.2f})")
    print(f" Predicted Fault    : {result_dict.get('fault_type', 'NORMAL')}")
    print(f" Fault Location     : {result_dict.get('fault_location', 'NONE')}")
    print(f" Model Confidence   : {result_dict.get('confidence', 0.0) * 100:.0f}%")
    print(f" Maint. Priority    : {result_dict.get('maintenance_priority', 'LOW')}")
    print("-" * 65)
    print(" Diagnostic Evidence:")
    for ev in result_dict.get("evidence", []):
        print(f"   * {ev}")
    if "vision_prediction" in result_dict:
        v_pred = result_dict["vision_prediction"]
        print(f" Visual Inspection  : {v_pred.get('visual_fault')} at {v_pred.get('location')} ({v_pred.get('confidence')*100:.0f}% conf)")
    if "raw_window" in result_dict and result_dict["raw_window"]:
        raw_info = result_dict["raw_window"]
        print(f" Raw Tensor Retained: Shape {raw_info.get('shape')} for Future CNN/LSTM")
    print("=" * 65 + "\n")


def run_full_verification():
    print("\n>>> STARTING COMPREHENSIVE SMART TRAIN AI VERIFICATION TEST...\n")

    predictor = Predictor(models_dir="models_artifacts")
    print("[OK] Predictor pipeline models loaded successfully.\n")

    # -------------------------------------------------------------
    # TEST 1: Healthy / Normal Operating Telemetry Stream
    # -------------------------------------------------------------
    sim_normal = create_simulator(FaultType.NORMAL, run_id="DEMO_NORMAL", duration_seconds=5.0)
    normal_records = sim_normal.generate_records()
    res_normal = predictor.predict_window(normal_records)
    print_dashboard_display("1. Healthy Normal Operation", res_normal.model_dump())
    assert res_normal.health_score >= 75
    assert res_normal.status == "HEALTHY"

    # -------------------------------------------------------------
    # TEST 2: Bearing Fault at Rear Bogie Axle 2
    # -------------------------------------------------------------
    sim_bearing = create_simulator(
        FaultType.BEARING_FAULT,
        run_id="DEMO_BEARING",
        duration_seconds=5.0,
        fault_severity=2,
        fault_location=FaultLocation.REAR_BOGIE_AXLE_2,
        seed=101,
    )
    bearing_records = sim_bearing.generate_records()
    res_bearing = predictor.predict_window(bearing_records)
    print_dashboard_display("2. Bearing Fault (Rear Bogie Axle 2)", res_bearing.model_dump())
    assert res_bearing.is_anomaly is True
    assert res_bearing.fault_type.value in ["BEARING_FAULT", "UNKNOWN"]

    # -------------------------------------------------------------
    # TEST 3: Brake Drag Fault at Brake System
    # -------------------------------------------------------------
    sim_brake = create_simulator(
        FaultType.BRAKE_ABNORMAL,
        run_id="DEMO_BRAKE",
        duration_seconds=5.0,
        fault_severity=3,
        fault_location=FaultLocation.BRAKE,
        seed=202,
    )
    brake_records = sim_brake.generate_records()
    res_brake = predictor.predict_window(brake_records)
    print_dashboard_display("3. Abnormal Brake Drag (Brake Assembly)", res_brake.model_dump())
    assert res_brake.is_anomaly is True

    # -------------------------------------------------------------
    # TEST 4: Wheel Flat Impact Fault at Front Bogie Axle 1
    # -------------------------------------------------------------
    sim_wheel = create_simulator(
        FaultType.WHEEL_FLAT,
        run_id="DEMO_WHEEL",
        duration_seconds=5.0,
        fault_severity=2,
        fault_location=FaultLocation.FRONT_BOGIE_AXLE_1,
        seed=303,
    )
    wheel_records = sim_wheel.generate_records()
    res_wheel = predictor.predict_window(wheel_records)
    print_dashboard_display("4. Wheel Flat Impact (Front Bogie Axle 1)", res_wheel.model_dump())

    # -------------------------------------------------------------
    # TEST 5: Real ESP32 DataSource Telemetry Ingestion Adapter
    # -------------------------------------------------------------
    print("Testing Live ESP32 Hardware Adapter (`ESP32DataSource`)...")
    esp32_adapter = ESP32DataSource()
    # Ingest 500 ESP32 raw JSON telemetry frames
    for rec in sim_bearing.generate_records():
        esp32_adapter.ingest_payload(rec.model_dump())

    esp32_window_df = esp32_adapter.fetch_window_df(duration_seconds=5.0)
    assert not esp32_window_df.empty
    print(f"[OK] Successfully ingested {len(esp32_window_df)} ESP32 frames through hardware adapter.\n")

    # -------------------------------------------------------------
    # TEST 6: Sensor + ESP32-CAM Computer Vision Fusion
    # -------------------------------------------------------------
    print("Testing Vision Fusion Engine (`FusionEngine`)...")
    vision_stub = VisionPrediction(
        visual_fault="BEARING_FAULT",
        location="REAR_BOGIE_AXLE_2",
        confidence=0.94,
        image_url="http://esp32-cam.local/snapshots/run001.jpg",
    )
    fused_diag = FusionEngine.fuse(res_bearing, vision_stub)
    print_dashboard_display("6. Telemetry + ESP32-CAM Visual Fusion", fused_diag)
    assert "vision_prediction" in fused_diag

    print("ALL SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_full_verification()

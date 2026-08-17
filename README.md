# Smart Train AI Fault Detection & Localization System

A physical-prototype predictive maintenance and fault localization architecture for smart rolling-stock (trains). Built using modular physics-based synthetic sensor data simulation first, and designed for hardware replacement with ESP32 microcontrollers and sensor payloads without modifying the machine learning pipeline.

---

## 🏗️ Architecture Overview

```text
SENSOR DATA (Synthetic Simulator / ESP32 Telemetry)
       │
       ▼
DATA VALIDATION (Schema & Physical Bound Verification)
       │
       ▼
TIME SYNCHRONIZATION (Multi-Rate Resampling: IMU 100Hz, INA219 10Hz, DS18B20 1Hz, HX711 80Hz)
       │
       ▼
SLIDING WINDOWING (5s Windows, 50% Overlap, Preserves Raw 3D Temporal Tensor for DL)
       │
       ▼
FEATURE EXTRACTION (Time-Domain, FFT Frequency-Domain, Thermal & Electrical Derived Ratios)
       │
       ├──────────────────────────────┐
       ▼                              ▼
ANOMALY DETECTION              FAULT CLASSIFICATION
(Isolation Forest)             (XGBoost Multiclass)
       │                              │
       └──────────────┬───────────────┘
                      ▼
             FAULT LOCALIZATION (Physical Location Classifier)
                      │
                      ▼
             HEALTH SCORE ENGINE (0-100 Health Index, Status, Priority & Evidence)
                      │
                      ▼
             INFERENCE REST API / DASHBOARD (FastAPI Backend)
```

---

## ⚡ Key Features

1. **Hardware-Agnostic Adapter Pattern**: Decouples data ingestion via `DataSource` abstraction (`SyntheticDataSource` vs `ESP32DataSource`).
2. **Physics-Plausible Simulator**: Generates correlated vibration, thermal rise, electrical drag, speed hunting, and load dynamics for 8 fault classes across severity levels 0..3.
3. **Zero Data Leakage Training**: Dataset splits are strictly grouped by `run_id` to ensure sliding windows from the same physical run never overlap across training, validation, and testing sets.
4. **Future Deep Learning & Vision Interfaces**: Preserves raw 3D time-series matrices for 1D CNN / TCN / LSTM models and includes `FusionEngine` stub for ESP32-CAM visual inspection predictions.

---

## 📁 Folder Structure

```text
smart_train_ai/
│
├── data/
│   ├── raw/
│   └── synthetic/               # Generated multi-run sensor telemetry (CSV)
│
├── simulator/
│   ├── base.py                  # Abstract simulator interface
│   ├── normal.py                # Healthy operating signals
│   ├── wheel_fault.py           # Wheel flat periodic rotational shock generator
│   ├── axle_fault.py            # Axle misalignment wobble generator
│   ├── bearing_fault.py         # Bearing high-frequency chatter & thermal rise
│   ├── brake_fault.py           # Mechanical drag & current surge generator
│   ├── suspension_fault.py      # Low-frequency bounce & vertical shock generator
│   ├── motor_fault.py           # Motor housing vibration & speed hunting generator
│   ├── body_damage.py           # Structural chassis resonance generator
│   └── generator.py             # Dataset generator CLI factory
│
├── preprocessing/
│   ├── validation.py            # Data bounds & schema validation
│   ├── synchronization.py       # Timestamp resampling & synchronization
│   └── windowing.py             # Sliding window extraction (tabular & raw 3D tensor)
│
├── features/
│   ├── time_domain.py           # RMS, Kurtosis, Crest Factor, Peak-to-Peak
│   ├── frequency_domain.py      # FFT Dominant Freq, Centroid, Band Energies
│   └── extractor.py             # Master Feature Extractor
│
├── models/
│   ├── anomaly.py               # Isolation Forest Anomaly Detector wrapper
│   ├── classifier.py            # XGBoost Multiclass Fault Classifier
│   ├── localization.py         # XGBoost Fault Localizer
│   └── health.py                # Health Score Engine (0-100, Priority, Evidence)
│
├── training/
│   ├── train_anomaly.py         # Stage 1 training script
│   ├── train_classifier.py      # Stage 2 training script (run_id group split)
│   ├── train_localization.py   # Stage 3 training script (run_id group split)
│   └── evaluate.py              # Metrics evaluation & report plot generation
│
├── inference/
│   └── predictor.py             # Integrated end-to-end inference pipeline
│
├── api/
│   └── main.py                  # FastAPI REST Service (/predict, /batch-predict, /model-info)
│
├── adapters/
│   ├── base.py                  # DataSource abstract base class
│   ├── synthetic.py             # Synthetic simulator adapter
│   └── esp32.py                 # Live ESP32 JSON payload adapter
│
├── fusion/
│   └── fusion_engine.py         # Sensor + ESP32-CAM Vision Prediction Fusion
│
├── tests/                       # Complete Pytest unit & E2E integration test suite
├── models_artifacts/            # Saved model artifacts (.joblib)
├── reports/                     # Metrics reports & confusion matrix plots
├── config.yaml                  # Central pipeline configuration
└── requirements.txt             # Python package dependencies
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Telemetry Dataset
```bash
python generate_dataset.py --runs_per_class 10 --duration 30
```

### 3. Train Pipeline Models
```bash
python -m smart_train_ai.training.train_anomaly
python -m smart_train_ai.training.train_classifier
python -m smart_train_ai.training.train_localization
python -m smart_train_ai.training.evaluate
```

### 4. Run Pytest Suite
```bash
python -m pytest tests/
```

### 5. Launch FastAPI REST Service
```bash
uvicorn smart_train_ai.api.main:app --reload
```

---

## 📊 Sample Diagnostic JSON Output

`POST /predict` response sample:

```json
{
  "train_id": "TRAIN001",
  "is_anomaly": true,
  "anomaly_score": 0.82,
  "fault_type": "BEARING_FAULT",
  "fault_location": "REAR_BOGIE_AXLE_2",
  "confidence": 0.98,
  "health_score": 42,
  "status": "CRITICAL",
  "maintenance_priority": "HIGH",
  "evidence": [
    "Anomaly score elevated (0.82)",
    "Vibration RMS increased (2.15g)",
    "Bearing temperature rising rapidly (+0.42°C/s)",
    "Bearing temperature rise delta detected (14.2°C)",
    "Classified fault 'BEARING_FAULT' with 98% confidence"
  ],
  "raw_window": {
    "channels": ["imu_x", "imu_y", "imu_z", "motor_temperature", "bearing_temperature", "rpm", "voltage", "current", "power", "load"],
    "shape": [500, 10]
  }
}
```

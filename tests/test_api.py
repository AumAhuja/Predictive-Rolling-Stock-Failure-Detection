import pytest
from fastapi.testclient import TestClient
from smart_train_ai.api.main import app
from smart_train_ai.schema import FaultType
from smart_train_ai.simulator.generator import create_simulator

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_api_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "supported_faults" in data


def test_api_predict():
    sim = create_simulator(FaultType.BEARING_FAULT, run_id="API_TEST", duration_seconds=5.0)
    records = sim.generate_records()
    payload = [r.model_dump() for r in records]

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "health_score" in res
    assert "fault_type" in res
    assert "fault_location" in res
    assert "evidence" in res

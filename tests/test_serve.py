import pytest
from fastapi.testclient import TestClient

from src.serve import app

NORMAL_READING = {
    "temperature": 70.0,
    "vibration": 0.5,
    "pressure": 100.0,
    "rotation_speed": 1500.0,
}

ANOMALOUS_READING = {
    "temperature": 145.0,
    "vibration": 2.8,
    "pressure": 190.0,
    "rotation_speed": 2800.0,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_detect_returns_valid_response(client):
    response = client.post("/detect", json=NORMAL_READING)
    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert "threshold" in data


def test_anomalous_reading_has_lower_score(client):
    normal_score = client.post("/detect", json=NORMAL_READING).json()["anomaly_score"]
    anomalous_score = client.post("/detect", json=ANOMALOUS_READING).json()["anomaly_score"]
    assert anomalous_score < normal_score


def test_custom_threshold_flags_everything(client):
    response = client.post("/detect?threshold=1.0", json=NORMAL_READING)
    assert response.status_code == 200
    assert response.json()["is_anomaly"] is True


def test_batch_detect(client):
    response = client.post("/detect/batch", json=[NORMAL_READING, ANOMALOUS_READING])
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert "anomaly_count" in data
    assert len(data["results"]) == 2


def test_batch_detect_empty_returns_400(client):
    response = client.post("/detect/batch", json=[])
    assert response.status_code == 400


def test_get_recent_anomalies(client):
    client.post("/detect?threshold=1.0", json=NORMAL_READING)
    response = client.get("/anomalies")
    assert response.status_code == 200
    assert "anomalies" in response.json()


def test_logs_returns_list(client):
    response = client.get("/logs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_stats_structure(client):
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "anomaly_rate" in data
    assert "total_readings" in data
    assert "avg_anomaly_score" in data

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.serve import SensorReading, app

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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pressure", -5.0),
        ("rotation_speed", -1500.0),
        ("vibration", -0.5),
        ("temperature", -273.15),
        ("temperature", 5000.0),
    ],
)
def test_extreme_reading_is_scored_not_rejected(client, field, value):
    """A broken sensor is what the detector exists to flag, so it must be scored."""
    response = client.post("/detect", json={**NORMAL_READING, field: value})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["anomaly_score"], float)
    assert isinstance(data["is_anomaly"], bool)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -300.0),  # below absolute zero: not physically representable
        ("pressure", "not-a-number"),
        ("vibration", None),
    ],
)
def test_invalid_reading_returns_422(client, field, value):
    response = client.post("/detect", json={**NORMAL_READING, field: value})
    assert response.status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "field", ["temperature", "vibration", "pressure", "rotation_speed"]
)
def test_non_finite_reading_is_rejected(field, value):
    """NaN/inf carry no signal and cannot be scored, so they are still rejected.

    Asserted against the model rather than the endpoint: a non-finite body never
    reaches the handler, and the 422 payload echoes the offending value, which is
    itself not JSON-encodable.
    """
    with pytest.raises(ValidationError):
        SensorReading(**{**NORMAL_READING, field: value})


def test_missing_field_returns_422(client):
    payload = {k: v for k, v in NORMAL_READING.items() if k != "pressure"}
    response = client.post("/detect", json=payload)
    assert response.status_code == 422


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

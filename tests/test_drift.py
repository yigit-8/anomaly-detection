"""Drift detection tests.

Both the reference file and the detection database are written under tmp_path,
so nothing here touches the repo's data/ directory.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src import drift


def make_reference(rng, n: int = 120) -> pd.DataFrame:
    """Readings from a machine running the way train.py's generator calls normal."""
    return pd.DataFrame({
        "temperature": rng.normal(70.0, 5.0, n),
        "vibration": rng.normal(0.5, 0.1, n),
        "pressure": rng.normal(100.0, 8.0, n),
        "rotation_speed": rng.normal(1500.0, 50.0, n),
    })


def make_shifted(rng, n: int = 120) -> pd.DataFrame:
    """The same machine running hot, loud and fast."""
    return pd.DataFrame({
        "temperature": rng.normal(130.0, 5.0, n),
        "vibration": rng.normal(2.2, 0.2, n),
        "pressure": rng.normal(180.0, 8.0, n),
        "rotation_speed": rng.normal(2600.0, 60.0, n),
    })


def write_current(db_path: str, frame: pd.DataFrame) -> None:
    """Persist rows the way serve.py's log_detection would have."""
    columns = ", ".join(f"{name} REAL" for name in drift.FEATURES)
    placeholders = ", ".join("?" for _ in drift.FEATURES)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            f"""CREATE TABLE detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {columns},
                    is_anomaly INTEGER,
                    anomaly_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )"""
        )
        conn.executemany(
            f"INSERT INTO detections ({', '.join(drift.FEATURES)}) VALUES ({placeholders})",
            frame[drift.FEATURES].itertuples(index=False, name=None),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def paths(tmp_path, monkeypatch):
    reference = tmp_path / "reference.csv"
    database = tmp_path / "predictions.db"
    report = tmp_path / "drift_report.html"
    monkeypatch.setattr(drift, "REFERENCE_PATH", str(reference))
    monkeypatch.setattr(drift, "DB_PATH", str(database))
    monkeypatch.setattr(drift, "REPORT_PATH", str(report))
    return {"reference": reference, "database": database, "report": report}


def test_drift_detected_when_readings_shift(paths):
    rng = np.random.default_rng(0)
    make_reference(rng).to_csv(paths["reference"], index=False)
    write_current(str(paths["database"]), make_shifted(rng))

    result = drift.run_drift_report()

    assert result["drift_detected"]
    assert result["current_rows"] == 120
    assert paths["report"].exists()


def test_no_drift_when_current_matches_reference(paths):
    rng = np.random.default_rng(0)
    reference = make_reference(rng)
    reference.to_csv(paths["reference"], index=False)
    write_current(str(paths["database"]), reference.copy())

    result = drift.run_drift_report()

    assert not result["drift_detected"]


def test_missing_reference_is_reported_not_raised(paths):
    result = drift.run_drift_report()
    assert result == {"drift_detected": False, "reason": "no_reference_data"}


def test_too_few_rows_is_reported_not_raised(paths):
    rng = np.random.default_rng(0)
    make_reference(rng).to_csv(paths["reference"], index=False)
    write_current(str(paths["database"]), make_shifted(rng, n=5))

    result = drift.run_drift_report()

    assert result == {"drift_detected": False, "reason": "insufficient_data"}

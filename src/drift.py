"""
Data drift detection for the anomaly detection pipeline.

Usage:
    python src/drift.py
"""

import os
import sqlite3

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "reference.csv")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "predictions.db")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "drift_report.html")

FEATURES = ["temperature", "vibration", "pressure", "rotation_speed"]


def load_reference() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_PATH)


def load_current(limit: int = 200) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=FEATURES)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        f"SELECT {', '.join(FEATURES)} FROM detections ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=FEATURES)


def run_drift_report(min_samples: int = 10) -> dict:
    if not os.path.exists(REFERENCE_PATH):
        print("No reference data. Run src/train.py first.")
        return {"drift_detected": False, "reason": "no_reference_data"}

    reference = load_reference()
    current = load_current()

    if len(current) < min_samples:
        print(f"Not enough data ({len(current)} rows, need {min_samples}).")
        return {"drift_detected": False, "reason": "insufficient_data"}

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report.save_html(REPORT_PATH)

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]

    print(f"Drift detected: {drift_detected}")
    print(f"Report saved to {REPORT_PATH}")
    return {"drift_detected": drift_detected, "current_rows": len(current)}


if __name__ == "__main__":
    run_drift_report()

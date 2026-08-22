"""
Anomaly detection API.

Loads the IsolationForest model saved by train.py and
classifies incoming sensor readings as normal or anomalous.
"""

import contextlib
import os
import sqlite3
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model.joblib")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "predictions.db")

model = None


def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found. Run src/train.py first.")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded.")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature     REAL,
            vibration       REAL,
            pressure        REAL,
            rotation_speed  REAL,
            is_anomaly      INTEGER,
            anomaly_score   REAL,
            timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def log_detection(features: dict, is_anomaly: bool, score: float):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """INSERT INTO detections
               (temperature, vibration, pressure, rotation_speed, is_anomaly, anomaly_score)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (features["temperature"], features["vibration"],
             features["pressure"], features["rotation_speed"],
             int(is_anomaly), score),
        )
        conn.commit()
    finally:
        conn.close()


def run_detection(reading_dict: dict, threshold: float) -> dict:
    df = pd.DataFrame([reading_dict])
    score = float(model.score_samples(df)[0])
    # IsolationForest's default decision boundary is 0; scores below threshold are anomalies
    is_anomaly = score < threshold
    return {"is_anomaly": is_anomaly, "anomaly_score": round(score, 4)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_model()
    yield


app = FastAPI(
    title="Anomaly Detection API",
    description="Detects anomalies in sensor readings using IsolationForest.",
    version="1.0.0",
    lifespan=lifespan,
)


class SensorReading(BaseModel):
    """A single sensor reading.

    The bounds below are sanity limits, not operating ranges. A failed sensor
    reporting a negative pressure or an absurd temperature is what this service
    exists to flag, so extreme readings are accepted and scored rather than
    rejected. Only input that cannot be scored at all is refused: non-numeric
    values, missing fields, NaN or infinity (``allow_inf_nan=False``) and
    temperatures below absolute zero.
    """

    temperature: float = Field(
        ...,
        ge=-273.15,
        le=1e6,
        allow_inf_nan=False,
        description="Temperature in Celsius (absolute zero is the only lower limit)",
    )
    vibration: float = Field(
        ...,
        allow_inf_nan=False,
        description="Vibration amplitude (unbounded; negative values reach the model)",
    )
    pressure: float = Field(
        ...,
        allow_inf_nan=False,
        description="Pressure in bar (unbounded; a vacuum or negative reading reaches the model)",
    )
    rotation_speed: float = Field(
        ...,
        allow_inf_nan=False,
        description="Rotation speed in RPM (unbounded; reversed rotation reaches the model)",
    )


class DetectionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    threshold: float


class BatchDetectionResponse(BaseModel):
    results: list[DetectionResponse]
    total: int
    anomaly_count: int


@app.get("/")
def root():
    return {"message": "Anomaly Detection API is running. Visit /docs for usage."}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "ok", "model_loaded": True}


@app.post("/detect", response_model=DetectionResponse)
def detect(
    reading: SensorReading,
    threshold: float = Query(
        default=0.0,
        description="Score threshold. Readings below this are flagged as anomalies.",
    ),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    result = run_detection(reading.model_dump(), threshold)
    log_detection(reading.model_dump(), result["is_anomaly"], result["anomaly_score"])
    return DetectionResponse(**result, threshold=threshold)


@app.post("/detect/batch", response_model=BatchDetectionResponse)
def detect_batch(
    readings: list[SensorReading],
    threshold: float = Query(default=0.0),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not readings:
        raise HTTPException(status_code=400, detail="Readings list cannot be empty.")

    results = []
    for reading in readings:
        result = run_detection(reading.model_dump(), threshold)
        log_detection(reading.model_dump(), result["is_anomaly"], result["anomaly_score"])
        results.append(DetectionResponse(**result, threshold=threshold))

    anomaly_count = sum(1 for r in results if r.is_anomaly)
    return BatchDetectionResponse(results=results, total=len(results), anomaly_count=anomaly_count)


@app.get("/anomalies")
def get_recent_anomalies(limit: int = Query(default=20, ge=1, le=500)):
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            """SELECT temperature, vibration, pressure, rotation_speed,
                      anomaly_score, timestamp
               FROM detections WHERE is_anomaly = 1
               ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    keys = ["temperature", "vibration", "pressure", "rotation_speed", "anomaly_score", "timestamp"]
    return {"anomalies": [dict(zip(keys, row)) for row in rows]}


@app.get("/logs")
def get_logs(limit: int = Query(default=20, ge=1, le=500)):
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            """SELECT temperature, vibration, pressure, rotation_speed,
                      is_anomaly, anomaly_score, timestamp
               FROM detections ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    keys = ["temperature", "vibration", "pressure", "rotation_speed",
            "is_anomaly", "anomaly_score", "timestamp"]
    return [dict(zip(keys, row)) for row in rows]


@app.get("/stats")
def get_stats():
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        anomalies = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE is_anomaly = 1"
        ).fetchone()[0]
        avg_score = conn.execute("SELECT AVG(anomaly_score) FROM detections").fetchone()[0]
    return {
        "total_readings": total,
        "anomaly_count": anomalies,
        "normal_count": total - anomalies,
        "anomaly_rate": round(anomalies / total, 4) if total > 0 else 0.0,
        "avg_anomaly_score": round(avg_score, 4) if avg_score else 0.0,
    }

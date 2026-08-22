"""
Anomaly detection training script.

Generates synthetic sensor readings, trains an IsolationForest model,
and logs everything to MLflow.

Usage:
    python src/train.py
    python src/train.py --contamination 0.05
"""

import argparse
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model.joblib")
REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "reference.csv")

FEATURES = ["temperature", "vibration", "pressure", "rotation_speed"]


def generate_data(n_samples: int, contamination: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies

    normal = pd.DataFrame({
        "temperature": rng.normal(70, 5, n_normal),
        "vibration": rng.normal(0.5, 0.1, n_normal),
        "pressure": rng.normal(100, 8, n_normal),
        "rotation_speed": rng.normal(1500, 50, n_normal),
        "label": 1,  # 1 = normal (IsolationForest convention)
    })

    anomalies = pd.DataFrame({
        "temperature": rng.uniform(110, 150, n_anomalies),
        "vibration": rng.uniform(1.5, 3.0, n_anomalies),
        "pressure": rng.uniform(160, 200, n_anomalies),
        "rotation_speed": rng.uniform(2200, 3000, n_anomalies),
        "label": -1,  # -1 = anomaly
    })

    return pd.concat([normal, anomalies], ignore_index=True).sample(frac=1, random_state=seed)


def train(n_samples: int, contamination: float, n_estimators: int) -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("anomaly-detection")

    with mlflow.start_run():
        mlflow.log_params({
            "n_samples": n_samples,
            "contamination": contamination,
            "n_estimators": n_estimators,
        })

        df = generate_data(n_samples, contamination)
        X = df[FEATURES]
        y_true = df["label"]

        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
        )
        model.fit(X)

        y_pred = model.predict(X)

        # IsolationForest returns -1 for anomalies, 1 for normal
        # Convert to binary for metric calculation: anomaly=1, normal=0
        y_true_binary = (y_true == -1).astype(int)
        y_pred_binary = (y_pred == -1).astype(int)

        metrics = {
            "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
            "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
            "anomaly_ratio": float(y_pred_binary.mean()),
        }
        mlflow.log_metrics(metrics)
        print(f"Metrics: {metrics}")

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        df[FEATURES].head(200).to_csv(REFERENCE_PATH, index=False)

        print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()
    train(args.n_samples, args.contamination, args.n_estimators)

"""
InsuSense - Glucose Spike Predictor

Owner: Samridh (ML + Data + Setup)
Responsibilities:
- Dataset loading/cleaning
- Feature engineering
- Model training + evaluation
- Saving artifacts to ./model/
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "data/data.csv"
MODEL_DIR = "model"
RANDOM_STATE = 42

FEATURES = ["carbs", "sugar", "fiber", "protein", "fat", "net_carbs"]


# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["net_carbs"] = df["carbs"] - df["fiber"]
    return df


# -------------------------------
# LABEL GENERATION (IMPROVED)
# -------------------------------
def generate_labels(df: pd.DataFrame) -> pd.Series:
    """
    Generates realistic-ish glucose spike labels.
    """

    net_carbs = df["carbs"] - df["fiber"]

    spike_score = (
        0.6 * net_carbs +
        0.3 * df["sugar"] -
        0.25 * df["fiber"] -
        0.1 * df["protein"] -
        0.05 * df["fat"]
    )

    # Add controlled noise for realism
    noise = np.random.normal(0, 5, size=len(df))
    spike_score += noise

    # Convert to categorical classes
    labels = []
    for val in spike_score:
        if val < 20:
            labels.append("Low")
        elif val < 45:
            labels.append("Medium")
        else:
            labels.append("High")

    return pd.Series(labels)


# -------------------------------
# LOAD DATA
# -------------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    required_cols = ["carbs", "sugar", "fiber", "protein", "fat"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Dataset missing required columns")

    return df


# -------------------------------
# TRAIN MODEL
# -------------------------------
def train(df: pd.DataFrame) -> Tuple[RandomForestClassifier, StandardScaler]:
    df = engineer_features(df)

    X = df[FEATURES]
    y = generate_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=RANDOM_STATE
    )

    model.fit(X_train_scaled, y_train)

    # -------------------------------
    # EVALUATION
    # -------------------------------
    y_pred = model.predict(X_test_scaled)

    print("\n📊 Model Evaluation")
    print("-" * 30)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, scaler


# -------------------------------
# SAVE ARTIFACTS
# -------------------------------
def save(model, scaler):
    Path(MODEL_DIR).mkdir(exist_ok=True)

    joblib.dump(model, f"{MODEL_DIR}/model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    print(f"\n✅ Model saved to {MODEL_DIR}/")


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("🚀 Training InsuSense model...")

    df = load_data(DATA_PATH)
    model, scaler = train(df)
    save(model, scaler)

    print("🎯 Training complete.")


if __name__ == "__main__":
    main()

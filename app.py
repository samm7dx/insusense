"""
InsuSense - Streamlit App (Collaborative ML Project)

Owner: Satwik (Backend + Frontend)
Responsibilities:
- Streamlit UI
- Prediction pipeline
- Loading artifacts from ./model/
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# -----------------------------
# CRITICAL: Shared feature code
# Must be identical in train_model.py
# -----------------------------
def engineer_features(df):
    df["net_carbs"] = df["carbs"] - df["fiber"]
    return df


FEATURES = ["carbs", "sugar", "fiber", "protein", "fat", "net_carbs"]


def _load_artifacts(model_dir: Path):
    model_path = model_dir / "model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Missing model artifacts. Run `python train_model.py` first to generate:\n"
            f"- {model_path}\n"
            f"- {scaler_path}"
        )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def _advice_for(pred_label: str) -> str:
    pred_label = str(pred_label)
    if pred_label.lower() == "high":
        return "High spike risk: reduce sugar and add protein."
    if pred_label.lower() == "low":
        return "Low spike risk: good choice."
    return "Medium spike risk: consider adding fiber/protein and reducing added sugars."


def main() -> None:
    st.set_page_config(page_title="InsuSense", page_icon="🩸", layout="centered")
    st.title("InsuSense — Glucose Spike Predictor(UI Test)")
    st.caption("Predicts glucose spike class from meal macros (Low / Medium / High).")

    repo_root = Path(__file__).resolve().parent
    model_dir = repo_root / "model"

    with st.sidebar:
        st.header("Inputs")
        carbs = st.number_input("Carbs (g)", min_value=0.0, value=50.0, step=1.0)
        sugar = st.number_input("Sugar (g)", min_value=0.0, value=15.0, step=1.0)
        fiber = st.number_input("Fiber (g)", min_value=0.0, value=6.0, step=1.0)
        protein = st.number_input("Protein (g)", min_value=0.0, value=15.0, step=1.0)
        fat = st.number_input("Fat (g)", min_value=0.0, value=10.0, step=1.0)
        predict_clicked = st.button("Predict", type="primary")

    if carbs < fiber:
        st.warning("Fiber cannot exceed carbs (otherwise net carbs becomes negative).")

    if not predict_clicked:
        st.info("Enter inputs on the left and click **Predict**.")
        return

    try:
        model, scaler = _load_artifacts(model_dir)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    input_df = pd.DataFrame(
        [
            {
                "carbs": carbs,
                "sugar": sugar,
                "fiber": fiber,
                "protein": protein,
                "fat": fat,
            }
        ]
    )
    input_df = engineer_features(input_df)
    X = input_df[FEATURES]

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    classes = list(model.classes_)
    conf = float(proba[classes.index(pred)])

    st.subheader("Result")
    st.metric("Prediction", str(pred), help="Low / Medium / High")
    st.metric("Confidence", f"{conf:.2%}", help="Model probability for the predicted class")

    st.subheader("Advice")
    st.write(_advice_for(str(pred)))

    with st.expander("Model details"):
        st.write("Feature order (must remain unchanged):")
        st.code(", ".join(FEATURES))
        st.write("Class probabilities:")
        st.json({c: float(p) for c, p in zip(classes, proba)})


if __name__ == "__main__":
    main()

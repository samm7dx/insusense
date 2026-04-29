#--OWNER- SATWIK RAJ

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


FEATURES = ["carbs", "sugar", "fiber", "protein", "fat", "net_carbs"]
RAW_FEATURES = ["carbs", "sugar", "fiber", "protein", "fat"]

BASELINE_GLUCOSE = 90.0
SAFE_THRESHOLD = 120.0
CAUTION_THRESHOLD = 160.0

PRESETS: Dict[str, Dict[str, float]] = {
    "Custom meal": {"carbs": 50.0, "sugar": 15.0, "fiber": 6.0, "protein": 15.0, "fat": 10.0},
    "White rice": {"carbs": 58.0, "sugar": 0.5, "fiber": 0.8, "protein": 4.0, "fat": 0.5},
    "Idli + sambar": {"carbs": 34.0, "sugar": 3.0, "fiber": 5.0, "protein": 8.0, "fat": 5.0},
    "Poha": {"carbs": 42.0, "sugar": 4.0, "fiber": 3.0, "protein": 6.0, "fat": 8.0},
    "Pizza slice": {"carbs": 34.0, "sugar": 4.0, "fiber": 2.0, "protein": 14.0, "fat": 12.0},
    "Salad bowl": {"carbs": 14.0, "sugar": 4.0, "fiber": 7.0, "protein": 6.0, "fat": 8.0},
    "Protein-heavy meal": {"carbs": 18.0, "sugar": 3.0, "fiber": 5.0, "protein": 32.0, "fat": 14.0},
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep identical feature engineering in app.py and train_model.py."""
    df = df.copy()
    df["net_carbs"] = df["carbs"] - df["fiber"]
    return df


@st.cache_resource(show_spinner=False)
def load_artifacts(model_dir: str) -> Tuple[object, object]:
    model_path = Path(model_dir) / "model.pkl"
    scaler_path = Path(model_dir) / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Missing model artifacts. Run `python train_model.py` first to create "
            f"{model_path.name} and {scaler_path.name}."
        )

    return joblib.load(model_path), joblib.load(scaler_path)


def safe_float(value: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def derive_profile_factor(
    weight: float,
    height_cm: float,
    body_fat_pct: float,
    age: int,
    sex: str,
) -> Dict[str, float]:
    height_m = max(height_cm / 100.0, 0.5)
    bmi = weight / (height_m ** 2)

    factor = 1.0
    if bmi >= 30:
        factor += 0.16
    elif bmi >= 25:
        factor += 0.08
    elif bmi < 18.5:
        factor -= 0.04

    if body_fat_pct >= 30:
        factor += 0.12
    elif body_fat_pct >= 25:
        factor += 0.06
    elif body_fat_pct < 12:
        factor -= 0.03

    if age >= 50:
        factor += 0.06
    elif age >= 35:
        factor += 0.03

    if sex.lower() == "female":
        factor *= 0.98

    factor = float(np.clip(factor, 0.78, 1.35))
    return {"bmi": bmi, "factor": factor}


def classify_risk(peak_glucose: float) -> Tuple[str, str]:
    if peak_glucose < SAFE_THRESHOLD:
        return "Safe", "success"
    if peak_glucose < CAUTION_THRESHOLD:
        return "Moderate", "warning"
    return "High", "error"


def meal_suggestions(
    carbs: float,
    sugar: float,
    fiber: float,
    protein: float,
    fat: float,
    net_carbs: float,
    risk_label: str,
    confidence: float,
) -> List[str]:
    suggestions: List[str] = []

    if sugar >= 15:
        suggestions.append("Reduce sugar: use less sweetener, fruit syrup, or dessert toppings.")
    if fiber < 5:
        suggestions.append("Increase fiber: add vegetables, legumes, oats, chia, or a salad side.")
    if protein < 10:
        suggestions.append("Increase protein: add eggs, paneer, curd, tofu, chicken, or dal.")
    if fat < 6 and carbs >= 35:
        suggestions.append("Add a small amount of healthy fat: nuts, seeds, or peanut butter can slow absorption.")
    if net_carbs >= 45:
        suggestions.append("Lower net carbs: reduce the carb portion and replace part of it with vegetables.")
    if carbs >= 60:
        suggestions.append("High carb load detected: split the meal or reduce rice/roti/pasta quantity.")

    if risk_label == "High":
        suggestions.append("This meal looks spike-prone. Pair it with fiber and protein, or choose a smaller portion.")
    elif risk_label == "Moderate":
        suggestions.append("This meal is borderline. A small fiber or protein boost can improve it.")
    else:
        suggestions.append("This meal is relatively balanced. Keep the portion controlled and stay active after eating.")

    if confidence < 0.6:
        suggestions.append("Model confidence is limited. Treat the result as a screening signal, not a diagnosis.")

    return list(dict.fromkeys(suggestions))


def explanatory_notes(
    carbs: float,
    sugar: float,
    fiber: float,
    protein: float,
    fat: float,
) -> List[str]:
    notes: List[str] = []
    if sugar >= max(8.0, carbs * 0.25):
        notes.append("Sugar is relatively high for the total carb load.")
    if fiber >= 6:
        notes.append("Fiber is helping reduce the expected spike.")
    if protein >= 15:
        notes.append("Protein should slow digestion and improve satiety.")
    if fat >= 12:
        notes.append("Fat may slow absorption, but large amounts also increase calorie density.")
    if carbs - fiber >= 40:
        notes.append("Net carbs are high enough to warrant portion control.")
    return notes


def build_illustrative_curve(
    carbs: float,
    sugar: float,
    fiber: float,
    protein: float,
    fat: float,
    profile_factor: float,
    model_label: str,
    confidence: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    net_carbs = max(carbs - fiber, 0.0)
    base_rise = (
        0.55 * net_carbs
        + 0.35 * sugar
        - 0.25 * fiber
        - 0.15 * protein
        - 0.08 * fat
    )

    label_multiplier = {"Low": 0.78, "Medium": 1.0, "High": 1.22}.get(model_label, 1.0)
    conf_multiplier = 0.85 + 0.3 * float(np.clip(confidence, 0.0, 1.0))
    spike = max(0.0, base_rise * profile_factor * label_multiplier * conf_multiplier)

    times = np.array([0, 30, 60, 90, 120], dtype=float)
    shape = np.array([0.0, 0.48, 1.0, 0.7, 0.28], dtype=float)
    curve = BASELINE_GLUCOSE + spike * shape
    return times, curve, float(curve.max())


def render_probability_table(probabilities: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Class": cls, "Probability": prob, "Display": f"{prob:.1%}"}
            for cls, prob in probabilities.items()
        ]
    )


def main() -> None:
    st.set_page_config(page_title="InsuSense", page_icon="🩸", layout="wide")

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem;}
        .insu-card {
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: rgba(250, 250, 250, 0.8);
        }
        .small-label {font-size: 0.82rem; color: #6b7280; margin-bottom: 0.2rem;}
        .big-value {font-size: 1.6rem; font-weight: 700; line-height: 1.1;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("InsuSense")
    st.caption("Meal-level glucose spike screening from macros with actionable suggestions.")

    repo_root = Path(__file__).resolve().parent
    model_dir = repo_root / "model"

    with st.sidebar:
        st.header("Personal profile")
        sex = st.selectbox("Sex", ["Male", "Female"], index=0)
        age = st.slider("Age", 10, 90, 24)
        weight = st.slider("Weight (kg)", 30.0, 180.0, 70.0, 0.5)
        height = st.slider("Height (cm)", 120.0, 220.0, 170.0, 0.5)
        body_fat = st.slider("Body fat (%)", 3.0, 50.0, 18.0, 0.5)

        st.divider()
        st.header("Meal preset")
        preset_name = st.selectbox("Choose a preset", list(PRESETS.keys()), index=0)
        st.caption("Selecting a preset only seeds the inputs. You can still edit every macro.")

    preset = PRESETS[preset_name]

    st.subheader("Enter food macros")
    with st.form("meal_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            carbs = st.number_input(
                "Carbs (g)",
                min_value=0.0,
                max_value=250.0,
                value=float(preset["carbs"]),
                step=0.1,
            )
            sugar = st.number_input(
                "Sugar (g)",
                min_value=0.0,
                max_value=150.0,
                value=float(preset["sugar"]),
                step=0.1,
            )

        with c2:
            fiber = st.number_input(
                "Fiber (g)",
                min_value=0.0,
                max_value=80.0,
                value=float(preset["fiber"]),
                step=0.1,
            )
            protein = st.number_input(
                "Protein (g)",
                min_value=0.0,
                max_value=120.0,
                value=float(preset["protein"]),
                step=0.1,
            )

        with c3:
            fat = st.number_input(
                "Fat (g)",
                min_value=0.0,
                max_value=120.0,
                value=float(preset["fat"]),
                step=0.1,
            )
            submitted = st.form_submit_button("Analyze meal", type="primary")

    if not submitted:
        st.info("Adjust the macro values and click **Analyze meal**.")
        return

    try:
        model, scaler = load_artifacts(str(model_dir))
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if carbs < fiber:
        st.warning("Fiber is greater than carbs. Net carbs will be negative, which is unusual for meal data.")

    input_df = pd.DataFrame(
        [
            {
                "carbs": safe_float(carbs),
                "sugar": safe_float(sugar),
                "fiber": safe_float(fiber),
                "protein": safe_float(protein),
                "fat": safe_float(fat),
            }
        ]
    )
    input_df = engineer_features(input_df)
    X = input_df[FEATURES]
    X_scaled = scaler.transform(X)

    pred_label = str(model.predict(X_scaled)[0])
    probabilities_arr = model.predict_proba(X_scaled)[0]
    classes = list(model.classes_)
    probabilities = {cls: float(prob) for cls, prob in zip(classes, probabilities_arr)}
    confidence = float(probabilities.get(pred_label, max(probabilities.values())))

    net_carbs = float(input_df.loc[0, "net_carbs"])
    profile = derive_profile_factor(
        weight=weight,
        height_cm=height,
        body_fat_pct=body_fat,
        age=age,
        sex=sex,
    )
    bmi = profile["bmi"]
    profile_factor = profile["factor"]

    times, curve, peak = build_illustrative_curve(
        carbs=carbs,
        sugar=sugar,
        fiber=fiber,
        protein=protein,
        fat=fat,
        profile_factor=profile_factor,
        model_label=pred_label,
        confidence=confidence,
    )
    risk_label, risk_kind = classify_risk(peak)
    suggestions = meal_suggestions(carbs, sugar, fiber, protein, fat, net_carbs, pred_label, confidence)
    notes = explanatory_notes(carbs, sugar, fiber, protein, fat)

    st.subheader("Summary")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(
            f'<div class="insu-card"><div class="small-label">Prediction</div><div class="big-value">{pred_label}</div></div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f'<div class="insu-card"><div class="small-label">Confidence</div><div class="big-value">{confidence:.1%}</div></div>',
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            f'<div class="insu-card"><div class="small-label">Illustrative peak</div><div class="big-value">{peak:.1f} mg/dL</div></div>',
            unsafe_allow_html=True,
        )
    with col_d:
        st.markdown(
            f'<div class="insu-card"><div class="small-label">Risk status</div><div class="big-value">{risk_label}</div></div>',
            unsafe_allow_html=True,
        )

    st.write(
        f"Estimated BMI: **{bmi:.1f}** | Net carbs: **{net_carbs:.1f} g** | Profile factor: **{profile_factor:.2f}**"
    )

    st.subheader("Glucose response visual")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, curve, marker="o")
    ax.axhline(SAFE_THRESHOLD, linestyle="--")
    ax.axhline(CAUTION_THRESHOLD, linestyle=":")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Glucose (illustrative mg/dL)")
    ax.set_title("Predicted post-meal glucose curve")
    ax.set_xticks(times)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, clear_figure=True)
    st.caption(
        "This chart is an illustrative risk curve generated from the meal profile and model output, "
        "not a medical glucose reading."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("What to improve")
        for item in suggestions:
            st.markdown(f"- {item}")

    with right:
        st.subheader("Why this result")
        if notes:
            for item in notes:
                st.markdown(f"- {item}")
        else:
            st.markdown("- No strong red flags from the macro pattern.")

    st.subheader("Class probabilities")
    st.dataframe(render_probability_table(probabilities), use_container_width=True, hide_index=True)

    with st.expander("Feature details"):
        st.write("Feature order used by the model:")
        st.code(", ".join(FEATURES))
        st.write("Engineered input row:")
        st.dataframe(input_df[FEATURES], use_container_width=True, hide_index=True)
        st.write(
            "Personal profile inputs are used for the advisory curve and suggestions, "
            "not for the current saved classifier."
        )


if __name__ == "__main__":
    main()

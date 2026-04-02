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

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


# -----------------------------
# CRITICAL: Shared feature code
# Must be identical in app.py
# -----------------------------
def engineer_features(df):
    df["net_carbs"] = df["carbs"] - df["fiber"]
    return df


FEATURES = ["carbs", "sugar", "fiber", "protein", "fat", "net_carbs"]

RAW_COLUMNS = ["carbs", "sugar", "fiber", "protein", "fat"]

INDIAN_FOOD_COLUMN_MAP = {
    "Carbohydrates (g)": "carbs",
    "Free Sugar (g)": "sugar",
    "Fibre (g)": "fiber",
    "Protein (g)": "protein",
    "Fats (g)": "fat",
}


def _load_and_clean_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Expected: data/data.csv"
        )

    df = pd.read_csv(csv_path)

    # Accept either:
    # - canonical schema: carbs,sugar,fiber,protein,fat
    # - Indian food schema: Carbohydrates (g), Free Sugar (g), Fibre (g), Protein (g), Fats (g)
    if set(RAW_COLUMNS).issubset(df.columns):
        df = df.copy()
    elif set(INDIAN_FOOD_COLUMN_MAP.keys()).issubset(df.columns):
        df = df.rename(columns=INDIAN_FOOD_COLUMN_MAP).copy()
    else:
        needed_a = set(RAW_COLUMNS)
        needed_b = set(INDIAN_FOOD_COLUMN_MAP.keys())
        raise ValueError(
            "Dataset columns do not match expected schemas.\n"
            f"- Expected either: {sorted(needed_a)}\n"
            f"- Or: {sorted(needed_b)}"
        )

    # Coerce numeric columns; handle missing safely
    for col in RAW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=RAW_COLUMNS).copy()

    # Clamp to reasonable domain (non-negative macros)
    for col in RAW_COLUMNS:
        df[col] = df[col].clip(lower=0)

    return df


def _create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create labels when the dataset has no ground-truth labels.

    Spike score:
      spike_score = (carbs + sugar) / (fiber + 1)

    Labels:
      Low    -> score < 10
      Medium -> 10-20 (inclusive)
      High   -> score > 20
    """
    spike_score = (df["carbs"] + df["sugar"]) / (df["fiber"] + 1.0)
    df = df.copy()
    df["spike_score"] = spike_score

    # Medium is inclusive for both ends per spec (10–20)
    conditions = [
        df["spike_score"] < 10,
        (df["spike_score"] >= 10) & (df["spike_score"] <= 20),
        df["spike_score"] > 20,
    ]
    choices = ["Low", "Medium", "High"]
    df["label"] = np.select(conditions, choices, default="Medium")
    return df


def _build_canonical_dataset_from_raw_indian_food(
    raw_path: Path, out_path: Path, *, per_class: int = 120, random_state: int = 42
) -> Path:
    """
    Build the canonical training CSV at `out_path` with columns:
      carbs,sugar,fiber,protein,fat

    Source: the Indian food nutrition dataset (kept in-repo for reproducibility).
    We *derive* labels during training, but we still keep `data/data.csv` label-free.
    """
    # Reuse loader logic (schema detection + numeric coercion)
    df = _load_and_clean_dataset(raw_path)

    labeled = _create_labels(engineer_features(df))

    # Balanced sampling across derived labels (when possible)
    groups = labeled.groupby("label", group_keys=False)
    sampled_parts: list[pd.DataFrame] = []
    for label, grp in groups:
        n = min(per_class, len(grp))
        if n == 0:
            continue
        sampled_parts.append(grp.sample(n=n, random_state=random_state))

    if not sampled_parts:
        raise ValueError(f"No usable rows found in {raw_path}")

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled[RAW_COLUMNS].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, index=False)
    return out_path


def _ensure_canonical_dataset(repo_root: Path) -> Path:
    """
    Ensure `data/data.csv` exists in canonical schema (no labels).

    If `data/raw_indian_food.csv` exists, we rebuild a balanced canonical dataset
    from it to keep training reproducible across machines.
    """
    data_dir = repo_root / "data"
    canonical_path = data_dir / "data.csv"
    raw_indian_path = data_dir / "raw_indian_food.csv"

    if raw_indian_path.exists():
        return _build_canonical_dataset_from_raw_indian_food(
            raw_indian_path, canonical_path, per_class=120, random_state=42
        )

    if not canonical_path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Expected {canonical_path} "
            "or data/raw_indian_food.csv to generate it."
        )

    return canonical_path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    data_path = _ensure_canonical_dataset(repo_root)
    model_dir = repo_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    df = _load_and_clean_dataset(data_path)
    df = engineer_features(df)
    df = _create_labels(df)

    X = df[FEATURES].copy()
    y = df["label"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Scale features (kept as a separate named step so we can also persist scaler)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight=None,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Persist artifacts required by app.py
    scaler = pipeline.named_steps["preprocess"].named_transformers_["num"]
    joblib.dump(pipeline.named_steps["model"], model_dir / "model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    print("Saved artifacts to:")
    print(f"- {model_dir / 'model.pkl'}")
    print(f"- {model_dir / 'scaler.pkl'}")


if __name__ == "__main__":
    main()

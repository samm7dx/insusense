# InsuSense

InsuSense is a collaborative ML project that predicts **glucose spike risk** from meal macros and classifies it into **Low / Medium / High** using:
- Inputs: `carbs`, `sugar`, `fiber`, `protein`, `fat`
- Feature engineering: `net_carbs = carbs - fiber`
- Model: `RandomForestClassifier` + `StandardScaler`

## Project structure

```
insusense/
│
├── app.py
├── train_model.py
├── requirements.txt
├── setup.sh
├── README.md
├── .gitignore
├── data/
│   └── data.csv
└── model/
    ├── model.pkl
    └── scaler.pkl
```

## Team roles

- **Samridh (ML + Data + Setup)**: dataset, feature engineering, model training, `train_model.py`
- **Satwik (Backend + Frontend)**: Streamlit UI + prediction pipeline, `app.py`
- **Surya (API + DevOps + Cloud)**: GitHub/deployment setup, `setup.sh`, `README.md`

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model (creates `model/model.pkl` and `model/scaler.pkl`):

```bash
python train_model.py
```

Or via setup script:

```bash
bash setup.sh
```

## Run the app

```bash
streamlit run app.py
```

## Deploy on Render

This repo includes a `Dockerfile` + `render.yaml` for one-click Render deploy.

1. Push the repo to GitHub (already done for `samm7dx/insusense`).
2. In Render, create a **New Web Service** and connect the GitHub repo.
3. Render will detect the Docker setup automatically.
4. Deploy. Render sets `$PORT`; the container runs Streamlit on that port.

## Notes for collaborators

- **Feature engineering must remain identical** across `train_model.py` and `app.py`.
- **Feature order must not change**:
  `["carbs","sugar","fiber","protein","fat","net_carbs"]`

## Using the Indian food dataset

If you have `Indian_Food_Nutrition_Processed.csv`, place it at:
- `data/raw_indian_food.csv`

`train_model.py` will automatically:
- Map columns (`Carbohydrates (g)`, `Free Sugar (g)`, `Fibre (g)`, `Protein (g)`, `Fats (g)`)
- Rebuild a **balanced** canonical `data/data.csv` (without labels)
- Derive labels during training using the spike-score rule

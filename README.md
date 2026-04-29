🩺 InsuSense
Personalized Glucose Spike Prediction using Machine Learning

InsuSense is a machine learning-powered web application that predicts post-meal glucose spike risk based on food macronutrients and provides visual insights + actionable dietary suggestions.

🚀 Live Demo

👉 https://insusense.onrender.com

📌 Overview

Understanding how food impacts blood glucose is critical for preventing lifestyle diseases like diabetes. However, most people lack access to continuous glucose monitoring or actionable insights.

InsuSense solves this by:

Predicting glucose spike risk (Low / Medium / High)
Generating a glucose response curve
Providing smart suggestions to improve meals
Offering an interactive UI for real-time analysis
🧠 Features

✔️ Predict glucose spike risk
✔️ Visual glucose response curve (0–120 mins)
✔️ Personalized dietary suggestions
✔️ User profile influence (weight, body fat, age)
✔️ Preset + custom meal input
✔️ Clean interactive UI (Streamlit)
✔️ Cloud deployed

⚙️ Tech Stack
Language: Python
ML: Scikit-learn (RandomForestClassifier)
Data: Pandas, NumPy
Visualization: Matplotlib
Frontend: Streamlit
Deployment: Render
Containerization: Docker
🧪 How It Works
🔹 Input

User enters:

Carbs
Sugar
Fiber
Protein
Fat
🔹 Feature Engineering
net_carbs = carbs - fiber
🔹 Prediction Pipeline
Feature engineering
Data scaling (StandardScaler)
Model prediction (Random Forest)
Risk classification
🔹 Output
📊 Risk Level (Low / Medium / High)
📈 Glucose Curve
💡 Suggestions
📊 Model Details
Algorithm: RandomForestClassifier

Features:

carbs, sugar, fiber, protein, fat, net_carbs
Training:
Synthetic + processed dataset
Balanced class generation
Evaluation:
Classification report
Accuracy metrics
📈 Sample Output
Meal	Net Carbs	Prediction
White Rice	High	High Risk
Salad	Low	Low Risk
Protein Meal	Medium	Moderate
📁 Project Structure
insusense/
│
├── app.py               # Streamlit UI + prediction
├── train_model.py       # Model training pipeline
├── setup.sh             # Setup automation
├── requirements.txt
├── Dockerfile
├── render.yaml
│
├── data/
│   └── data.csv
│
└── model/
    ├── model.pkl
    └── scaler.pkl
⚙️ Installation
git clone https://github.com/samm7dx/insusense.git
cd insusense
python -m venv venv
source venv/Scripts/activate   # Windows
pip install -r requirements.txt
▶️ Run Locally
python -m streamlit run app.py

Open:

http://localhost:8501
🌐 Deployment
Hosted on Render
Uses Docker container
Automatically builds and runs via setup.sh
👥 Team
🔹 Samridh Raj (ML + Data + Setup)
Dataset preparation
Feature engineering
Model training
train_model.py
🔹 Satwik Raj (Backend + Frontend)
Streamlit UI
Prediction pipeline
Visualization
app.py
🔹 Surya Gautam (DevOps + Deployment)
GitHub setup
Docker + deployment
Automation scripts
setup.sh, README.md
⚠️ Disclaimer

This project is intended for educational purposes only.
It is not a medical tool and should not be used for diagnosis.

🚀 Future Improvements
Real CGM dataset integration
Regression-based glucose prediction
Mobile app version
User history tracking
API backend (FastAPI)
⭐ Contribute

Pull requests are welcome. For major changes, open an issue first.

📌 Key Insight

Food composition directly influences glucose response — InsuSense translates that into actionable intelligence.

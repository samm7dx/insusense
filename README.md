
InsuSense

Personalized Glucose Spike Prediction using Machine Learning


---

Overview

InsuSense is a machine learning-powered web application that predicts the risk of post-meal glucose spikes based on food macronutrients. It helps users understand how different foods impact their blood glucose levels and provides meaningful insights along with practical dietary suggestions.

The goal is to make glucose awareness simple, interactive, and accessible for everyday use.


---

Features

Predicts glucose spike risk (Low, Medium, High)

Generates a glucose response curve (0–120 minutes)

Provides personalized dietary recommendations

Supports both custom inputs and preset meals

Interactive user interface built with Streamlit

Cloud deployment using Render



---

How It Works

Input

The user provides macronutrient values:

Carbohydrates

Sugar

Fiber

Protein

Fat



---

Feature Engineering

Net carbohydrates are calculated to represent the actual glucose-impacting component:

net_carbs = carbs - fiber


---

Prediction Pipeline

1. Feature engineering


2. Data scaling using StandardScaler


3. Prediction using Random Forest Classifier


4. Classification into risk categories




---

Output

Risk level (Low / Medium / High)

Glucose response curve

Actionable dietary suggestions



---

Model Details

Algorithm: Random Forest Classifier

Features used:

Carbohydrates

Sugar

Fiber

Protein

Fat

Net carbohydrates



Approach

Synthetic dataset generation

Feature-based glucose spike scoring

Classification into defined risk levels



---

Sample Predictions

Meal Type	Net Carbs	Prediction

White Rice	High	High Risk
Salad	Low	Low Risk
Protein Meal	Medium	Medium Risk



---

Project Structure

insusense/
│
├── app.py              # Streamlit UI and prediction logic  
├── train_model.py      # Model training pipeline  
├── setup.sh            # Setup script  
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


---

Installation

git clone https://github.com/samm7dx/insusense.git
cd insusense

python -m venv venv
source venv/Scripts/activate   # Windows

pip install -r requirements.txt


---

Run Locally

python -m streamlit run app.py

Open in browser:
http://localhost:8501


---

Deployment

Platform: Render

Containerized using Docker

Accessible through a web browser



---

Team

Samridh Raj (Machine Learning and Data)

Dataset preparation

Feature engineering

Model training

train_model.py


Satwik Raj (Backend and Frontend)

Streamlit interface

Prediction pipeline

Data visualization

app.py


Surya Gautam (DevOps and Deployment)

GitHub setup

Deployment configuration

Docker and setup automation

Documentation



---

Disclaimer

This project is intended for educational purposes only.
It should not be used for medical diagnosis or treatment.


---

Future Improvements

Integration with real Continuous Glucose Monitoring (CGM) datasets

Regression-based glucose prediction

User personalization and history tracking

Mobile application development

API backend using FastAPI



---

Contribution

Contributions are welcome.
You can fork the repository and submit a pull request with improvements or new features.


---

Key Insight

Food composition directly affects glucose response.
InsuSense translates that relationship into clear, actionable insights.



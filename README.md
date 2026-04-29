# 🩺 InsuSense  
### Personalized Glucose Spike Prediction using Machine Learning  

---

## 📌 Overview  

InsuSense is a machine learning-based web application that predicts **post-meal glucose spike risk** using food macronutrients. It helps users understand how their dietary choices affect glucose levels and provides **visual insights along with actionable suggestions**.

The system is designed to be simple, interactive, and accessible, making it useful for everyday health awareness and preventive care.

---

## 🚀 Features  

- Predict glucose spike risk (**Low / Medium / High**)  
- Generate **glucose response curve (0–120 minutes)**  
- Provide **personalized dietary suggestions**  
- Support **custom and preset meals**  
- Interactive UI using Streamlit  
- Cloud deployment (Render)  

---

## 🧠 How It Works  

### 🔹 Input  
User provides macronutrient values:  
- Carbohydrates  
- Sugar  
- Fiber  
- Protein  
- Fat  

---

### 🔹 Feature Engineering  

Net carbohydrates represent the effective glucose-impacting component.
net_carbs = carbs - fiber
---

### 🔹 Prediction Pipeline  

1. Feature Engineering  
2. Data Scaling (StandardScaler)  
3. Random Forest Model  
4. Risk Classification  

---

### 🔹 Output  

- 📊 Risk Level (Low / Medium / High)  
- 📈 Glucose Curve  
- 💡 Suggestions  

---

## 📊 Model Details  

- Algorithm: RandomForestClassifier  
- Features Used:
  - carbs  
  - sugar  
  - fiber  
  - protein  
  - fat  
  - net_carbs  

- Approach:
  - Synthetic dataset generation  
  - Feature-based spike scoring  
  - Classification into risk levels  

---

## 📈 Sample Predictions  

| Meal Type     | Net Carbs | Prediction   |
|--------------|----------|-------------|
| White Rice   | High     | High Risk   |
| Salad        | Low      | Low Risk    |
| Protein Meal | Medium   | Medium Risk |

---

## 📁 Project Structure  
insusense/
│
├── app.py # Streamlit UI + prediction logic
├── train_model.py # Model training pipeline
├── setup.sh # Setup automation script
├── requirements.txt
├── Dockerfile
├── render.yaml
│
├── data/
│ └── data.csv
│
└── model/
├── model.pkl
└── scaler.pkl


## ⚙️ Installation  
git clone https://github.com/samm7dx/insusense.git

cd insusense
python -m venv venv
source venv/Scripts/activate # Windows
pip install -r requirements.txt
---
## ▶️ Run Locally  

python -m streamlit run app.py


Open in browser:  
http://localhost:8501  

---

## 🌐 Deployment  

- Platform: Render  
- Containerized using Docker  
- Accessible via browser  

---

## 👥 Team  

**Samridh Raj (ML + Data + Setup)**  
- Dataset preparation  
- Feature engineering  
- Model training  
- train_model.py  

**Satwik Raj (Backend + Frontend)**  
- Streamlit UI  
- Prediction pipeline  
- Visualization  
- app.py  

**Surya Gautam (DevOps + Deployment)**  
- GitHub setup  
- Deployment configuration  
- Docker + setup.sh  
- README  

---

## ⚠️ Disclaimer  

This project is for **educational purposes only**.  
It is not intended for medical diagnosis or treatment.

---

## 🚀 Future Improvements  

- Integration with real CGM datasets  
- Regression-based glucose prediction  
- User personalization and history tracking  
- Mobile app development  
- API backend using FastAPI  

---

## ⭐ Contribution  

Contributions are welcome.  
Feel free to fork the repository and submit a pull request.

---

## 📌 Key Insight  

Food composition directly influences glucose response —  
**InsuSense converts that into actionable insights.**


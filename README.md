<h1 align="center">🩺 HealthOracle - AI Powered Disease Prediction System</h1>

<p align="center">
  <img src="https://img.shields.io/github/languages/top/roshan-poudhyal/HealthOracle?color=blue&style=flat-square" />
  <img src="https://img.shields.io/github/repo-size/roshan-poudhyal/HealthOracle?style=flat-square" />
  <img src="https://img.shields.io/github/last-commit/roshan-poudhyal/HealthOracle?style=flat-square" />
  <img src="https://img.shields.io/github/issues/roshan-poudhyal/HealthOracle?style=flat-square" />
</p>

<p align="center">
  <img src="https://github.com/roshan-poudhyal/HealthOracle/assets/animated-ai-health.gif" alt="HealthOracle Animation" height="300"/>
</p>

---

## 🧠 About the Project

**HealthOracle** is a smart, AI-powered web-based diagnostic system built using **Django** and **TensorFlow Lite** that helps users predict the risk level of 4 major diseases:

- ❤️ Heart Disease
- 🫁 Lung Disease
- 🩸 Liver Disease
- 🍬 Diabetes

It also provides **personalized health advice** and risk categorization (Low / Moderate / High) based on user input and deep learning models.

---

## 🌟 Features

- 🔍 AI-based Risk Prediction (TFLite Models)
- 🧮 Feature Scaling using saved Scikit-learn scalers
- 🖼️ Modern UI with disease-specific forms
- 📈 Risk categories with personalized suggestions
- ⚡ Fast real-time predictions
- 🧪 Tested ML models with high accuracy

---

## 🧰 Tech Stack

| Category         | Tools Used                                   |
|------------------|----------------------------------------------|
| 💻 Backend       | Django, Python, SQLite                       |
| 🧠 AI/ML         | TensorFlow Lite, Scikit-learn (StandardScaler) |
| 📦 Deployment    | GitHub                                       |
| 🖼 Frontend      | HTML, CSS (via Django Templates)             |
| 📊 Data Storage  | `.tflite`, `.pkl` models, SQLite             |

---

## 🔎 Demo Preview (GIF)

> Here's a walkthrough of the full app:  
> *Input > Predict > Risk Level > Suggestion*

<img src="https://github.com/roshan-poudhyal/HealthOracle/assets/demo.gif" width="600"/>

---

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/roshan-poudhyal/HealthOracle.git
cd HealthOracle

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Django server
python manage.py runserver


HealthOracle/
│
├── templates/                # HTML templates
│   ├── home.html
│   ├── heart.html
│   ├── lung.html
│   ├── liver.html
│   └── diabetes.html
│
├── heart_model.tflite        # TFLite ML Models
├── lung_model.tflite
├── liver_model.tflite
├── diabetes_model.tflite
│
├── heart_scaler.pkl          # Preprocessing scalers
├── lung_scaler.pkl
├── liver_scaler.pkl
├── diabetes_scaler.pkl
│
├── views.py                  # Django view logic
├── urls.py                   # URL Routing
├── manage.py
└── requirements.txt

# Telco Customer Churn Prediction

An end to end machine learning project that predicts customer churn using structured telecom data.  
The project includes data preprocessing, model training, hyperparameter tuning, model evaluation, and deployment using FastAPI.

---

## 📌 Project Overview

This project aims to build a production ready churn prediction system using modern machine learning practices.  

It includes:

- Data cleaning and preprocessing
- Feature engineering
- Handling categorical variables using One Hot Encoding
- Model training with Logistic Regression, Random Forest, and XGBoost
- Cross validation and hyperparameter tuning
- Threshold optimization
- Model serialization
- REST API deployment using FastAPI

---

## 🎯 Problem Statement

Customer churn is a critical issue for subscription based businesses. Losing customers directly impacts revenue and growth. 

The goal of this project is to build a machine learning system that can predict whether a customer is likely to churn based on their demographic and service usage data, enabling proactive retention strategies.

---

## 📊 Dataset

- Source: Telco Customer Churn dataset
- Target variable: Churn (Yes / No)
- Converted to binary format: 0 and 1
- No missing values after preprocessing
- Final feature set includes numerical and categorical variables

---

## 🧠 Models Used

- Logistic Regression
- Random Forest
- XGBoost

Best performing model: XGBoost

Final Performance:

- ROC AUC: ~0.84
- Accuracy: ~0.78 to 0.80
- Threshold optimized using F1 score

---

## ⚙️ Pipeline Design

The model is built using an sklearn Pipeline including:

- ColumnTransformer
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- XGBoost classifier

This ensures clean, reproducible training and deployment.

---

## 📦 Project Structure

telco-churn-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── churn_xgb_model.pkl
│
├── notebooks/
│   └── 02_eda.ipynb
│
├── api/
│   └── main.py
│
├── .gitignore
├── requirements.txt
└── README.md

---

## 🚀 API Deployment

The trained model is deployed using FastAPI.

To run locally:

uvicorn api.main:app --reload

API documentation:

http://127.0.0.1:8000/docs


# 📥 Input Example

Send a POST request to:

/predict

With JSON body like this:

{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 1074.0
}

# 📤 Output Example

The API returns churn probability and prediction:

{
  "probability": 0.87,
  "prediction": 1
}

Where:

probability means likelihood of churn

prediction 0 means not churn

prediction 1 means churn

---

## 🛠 Technologies Used

Python

Pandas

NumPy

Scikit Learn

XGBoost

FastAPI

Uvicorn

Matplotlib

Seaborn

---

## 📈 Evaluation Strategy

Train Test Split

Stratified Cross Validation

ROC AUC scoring

Threshold optimization

Confusion matrix analysis

---

## 🎯 Future Improvements

Feature importance analysis

SHAP explainability

Docker containerization

Cloud deployment

Monitoring and logging

CI CD integration

---

## 👨‍💻 Author

Tahmeed Siraz
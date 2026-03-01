from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/churn_xgb_model.pkl")


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prob = model.predict_proba(df)[:, 1][0]

    return {
        "probability": float(prob),
        "prediction": int(prob > 0.35)
    }

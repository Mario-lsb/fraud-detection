from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="Fraud Detection API")

lr  = joblib.load("LogReg_model.pkl")
rf  = joblib.load("RandomForest_model.pkl")
iso = joblib.load("isolation_forest.pkl")

mse_scores = np.load("mse_scores.npy")
THRESHOLD  = np.percentile(mse_scores, 97)

class Transaction(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"status": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(tx: Transaction):
    x = np.array(tx.features).reshape(1, -1)

    lr_result  = int(lr.predict(x)[0])
    rf_result  = int(rf.predict(x)[0])
    iso_result = int(iso.predict(x)[0] == -1)

    rf_proba   = float(rf.predict_proba(x)[0][1])
    ae_result  = int(rf_proba > 0.7)   # using RF probability as AE proxy

    votes = lr_result + rf_result + iso_result + ae_result

    return {
        "fraud_detected": votes >= 2,
        "votes": f"{votes}/4 models flagged fraud",
        "logistic_regression": lr_result,
        "random_forest": rf_result,
        "isolation_forest": iso_result,
        "rf_fraud_probability": round(rf_proba * 100, 2)
    }
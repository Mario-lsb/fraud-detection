from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import sqlite3
from datetime import datetime

app = FastAPI(title="Fraud Detection API")

lr  = joblib.load("LogReg_model.pkl")
rf  = joblib.load("RandomForest_model.pkl")
iso = joblib.load("isolation_forest.pkl")
mse_scores = np.load("mse_scores.npy")
THRESHOLD  = np.percentile(mse_scores, 97)


# ── Database setup ─────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("predictions.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT,
            fraud_detected INTEGER,
            rf_probability REAL,
            lr_result      INTEGER,
            rf_result      INTEGER,
            iso_result     INTEGER,
            votes          INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()


def log_prediction(fraud_detected, rf_prob, lr_result, rf_result, iso_result, votes):
    conn = sqlite3.connect("predictions.db")
    conn.execute("""
        INSERT INTO predictions
            (timestamp, fraud_detected, rf_probability, lr_result, rf_result, iso_result, votes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        int(fraud_detected),
        rf_prob,
        lr_result,
        rf_result,
        iso_result,
        votes
    ))
    conn.commit()
    conn.close()


# ── Models ─────────────────────────────────────────────────────────────────────
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
    ae_result  = int(rf_proba > 0.7)

    votes       = lr_result + rf_result + iso_result + ae_result
    is_fraud    = votes >= 2

    # Log every prediction to database
    log_prediction(is_fraud, round(rf_proba * 100, 2), lr_result, rf_result, iso_result, votes)

    return {
        "fraud_detected":      is_fraud,
        "votes":               f"{votes}/4 models flagged fraud",
        "logistic_regression": lr_result,
        "random_forest":       rf_result,
        "isolation_forest":    iso_result,
        "rf_fraud_probability": round(rf_proba * 100, 2)
    }


@app.get("/stats")
def stats():
    """Returns aggregate stats from the persistent prediction log."""
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE fraud_detected = 1")
    fraud_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT timestamp, fraud_detected, rf_probability, votes
        FROM predictions
        ORDER BY id DESC
        LIMIT 50
    """)
    recent = [
        {"timestamp": r[0], "fraud": bool(r[1]), "rf_prob": r[2], "votes": r[3]}
        for r in cursor.fetchall()
    ]

    conn.close()

    return {
        "total_checked": total,
        "fraud_count":   fraud_count,
        "fraud_rate":    round((fraud_count / total * 100), 2) if total > 0 else 0,
        "recent":        recent
    }
import numpy as np
import requests
import time
import random

# Load test data
X_test = np.load("X_test.npy")
API_URL = "http://127.0.0.1:8000/predict"

print("=" * 50)
print("FraudShield — Real-Time Transaction Simulator")
print("=" * 50)
print(f"Loaded {len(X_test)} transactions from test set")
print("Sending transactions to API every 2 seconds...")
print("Press Ctrl+C to stop\n")

total    = 0
fraud    = 0

try:
    while True:
        # Pick a random transaction from test data
        idx      = random.randint(0, len(X_test) - 1)
        features = X_test[idx].tolist()

        try:
            res = requests.post(
                API_URL,
                json={"features": features},
                timeout=10
            ).json()

            total += 1
            is_fraud = res["fraud_detected"]
            if is_fraud:
                fraud += 1

            status = "🚨 FRAUD    " if is_fraud else "✅ Legit    "
            prob   = res["rf_fraud_probability"]
            rate   = round((fraud / total) * 100, 1)

            print(f"Tx #{total:04d} | Sample {idx:05d} | {status} | RF Prob: {prob:5.1f}% | Fraud Rate: {rate}%")

        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to API. Make sure uvicorn is running.")
            break
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(2)

except KeyboardInterrupt:
    print(f"\n{'='*50}")
    print(f"Simulation stopped.")
    print(f"Total transactions: {total}")
    print(f"Fraud detected:     {fraud}")
    print(f"Fraud rate:         {round((fraud/total)*100, 1) if total > 0 else 0}%")
    print(f"All predictions saved to predictions.db")
    print(f"{'='*50}")
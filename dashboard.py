import streamlit as st
import requests
import numpy as np

st.set_page_config(page_title="AI Fraud Detection", page_icon="🔍", layout="wide")
st.title("🔍 AI-Powered Credit Card Fraud Detection")
st.divider()

X_test = np.load("X_test.npy")
n_features = X_test.shape[1]

st.sidebar.header("⚙️ Options")
use_sample = st.sidebar.checkbox("Use sample from test data", value=True)

if use_sample:
    sample_index = st.sidebar.number_input(
        "Sample index", 
        min_value=0, 
        max_value=len(X_test)-1, 
        value=0, 
        step=1
    )
    features = X_test[int(sample_index)].tolist()
    st.info(f"Using test sample #{int(sample_index)}")
else:
    features = []
    cols = st.columns(5)
    for i in range(n_features):
        val = cols[i % 5].number_input(f"F{i+1}", value=0.0, key=f"f{i}")
        features.append(val)

if st.button("🚨 Analyse Transaction", use_container_width=True, type="primary"):
    with st.spinner("Running all models..."):
        try:
            res = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"features": features}
            ).json()

            st.divider()

            if res["fraud_detected"]:
                st.error(f"## 🚨 FRAUD DETECTED — {res['votes']}")
            else:
                st.success(f"## ✅ LEGITIMATE — {res['votes']}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Logistic Regression", "🚨 Fraud" if res.get("logistic_regression") else "✅ Legit")
            col2.metric("Random Forest",        "🚨 Fraud" if res.get("random_forest") else "✅ Legit")
            col3.metric("Isolation Forest",     "🚨 Fraud" if res.get("isolation_forest") else "✅ Legit")
            col4.metric("RF Fraud Probability", f"{res['rf_fraud_probability']}%")

        except Exception as e:
            st.error(f"Cannot connect to API: {e}")
            st.info("Make sure the API is running: python -m uvicorn app:app --reload"
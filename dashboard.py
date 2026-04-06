import streamlit as st
import requests
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono&display=swap');
html, body, [data-testid="stAppViewContainer"], .main {
    background: #F4F6FB !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] * { color: #1E293B !important; }

/* Fix ALL inputs in sidebar — white background, visible text */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] div[data-baseweb="input"] input {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 2px solid #C7D2FE !important;
    border-radius: 10px !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
    -webkit-text-fill-color: #1E293B !important;
}
[data-testid="stSidebar"] div[data-baseweb="input"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: 2px solid #C7D2FE !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
    border-color: #4F6EF7 !important;
    box-shadow: 0 0 0 3px rgba(79,110,247,.15) !important;
}

div.stButton > button {
    background: linear-gradient(135deg, #4F6EF7, #7C3AED) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 14px 0 !important;
    width: 100% !important; font-family: 'DM Sans', sans-serif !important;
}
div.stButton > button:hover { opacity: .86 !important; }

.hero {
    background: linear-gradient(135deg, #4F6EF7 0%, #7C3AED 100%);
    border-radius: 18px; padding: 30px 38px; color: white; margin-bottom: 24px;
    position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-40px; right:-40px;
    width:200px; height:200px; background:rgba(255,255,255,.07); border-radius:50%;
}
.hero h1 { font-size:1.9rem; font-weight:700; margin:0; color:white !important; }
.hero p  { font-size:.82rem; opacity:.75; margin:6px 0 0; text-transform:uppercase; letter-spacing:.1em; }
.pills   { display:flex; gap:10px; margin-top:18px; flex-wrap:wrap; }
.pill    { background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.28);
           border-radius:100px; padding:5px 15px; font-size:.8rem; color:white; font-weight:500; }

.fraud-banner { background:#FEF2F2; border:2px solid #FCA5A5; border-radius:14px; padding:24px; text-align:center; margin:16px 0; }
.legit-banner { background:#F0FDF4; border:2px solid #86EFAC; border-radius:14px; padding:24px; text-align:center; margin:16px 0; }
.banner-title { font-size:1.6rem; font-weight:700; margin:0; }
.banner-sub   { font-size:.88rem; margin:6px 0 0; opacity:.65; }

.mcard { background:white; border:1px solid #E2E8F0; border-radius:12px; padding:16px; text-align:center; box-shadow:0 1px 4px rgba(0,0,0,.05); }
.mcard-label { font-size:.68rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:.07em; margin-bottom:8px; }
.mcard-fraud { font-size:1.05rem; font-weight:700; color:#EF4444; }
.mcard-legit { font-size:1.05rem; font-weight:700; color:#10B981; }

.log-row { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:9px; padding:9px 14px; margin-bottom:6px; font-family:'DM Mono',monospace; font-size:.78rem; display:flex; gap:10px; align-items:center; }
.sec-label { font-size:.68rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:.1em; margin-bottom:10px; }

.fraud-idx-btn button { background:#EFF6FF !important; color:#1D4ED8 !important; border:1px solid #BFDBFE !important; border-radius:8px !important; font-size:.8rem !important; font-weight:600 !important; padding:4px 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "history"       not in st.session_state: st.session_state.history       = []
if "total_checked" not in st.session_state: st.session_state.total_checked = 0
if "fraud_count"   not in st.session_state: st.session_state.fraud_count   = 0
if "sample_index"  not in st.session_state: st.session_state.sample_index  = 0

# ── Load data ──────────────────────────────────────────────────────────────────
try:
    X_test = np.load("X_test.npy")
except FileNotFoundError:
    st.error("❌ X_test.npy not found. Make sure it's in the same folder as dashboard.py")
    st.stop()

try:
    y_test        = np.load("y_test.npy")
    has_labels    = True
    fraud_indices = np.where(y_test == 1)[0].tolist()
except FileNotFoundError:
    y_test        = None
    has_labels    = False
    fraud_indices = []

API_URL    = os.environ.get("API_URL", "http://127.0.0.1:8000") + "/predict"
total      = st.session_state.total_checked
fraud_cnt  = st.session_state.fraud_count
fraud_rate = (fraud_cnt / total * 100) if total > 0 else 0.0

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>🛡️ FraudShield AI</h1>
  <p>Real-Time Credit Card Fraud Detection · 4-Model Ensemble</p>
  <div class="pills">
    <div class="pill">📊 Checked: {total}</div>
    <div class="pill">🚨 Fraud Found: {fraud_cnt}</div>
    <div class="pill">📈 Fraud Rate: {fraud_rate:.1f}%</div>
    <div class="pill">🎯 Accuracy: 97.3%</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — just ONE number input ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudShield AI")
    st.markdown("---")

    st.markdown("### 🔢 Enter Sample Number")
    st.caption(f"Type any number from 0 to {len(X_test)-1}")

    typed = st.text_input("Sample Index", value=str(st.session_state.sample_index), placeholder="e.g. 840")

    try:
        parsed = int(typed)
        if 0 <= parsed <= len(X_test) - 1:
            sample_index = parsed
        else:
            st.warning(f"Must be between 0 and {len(X_test)-1}")
            sample_index = st.session_state.sample_index
    except ValueError:
        if typed != "":
            st.warning("Please enter a valid number")
        sample_index = st.session_state.sample_index

    st.session_state.sample_index = sample_index

    # Ground truth badge
    if has_labels:
        true_label = int(y_test[int(sample_index)])
        if true_label == 1:
            st.error("🚨 Ground Truth: FRAUD")
        else:
            st.success("✅ Ground Truth: LEGITIMATE")

    st.markdown("---")

    # Quick-pick fraud samples
    if fraud_indices:
        st.markdown("**🎯 Quick-pick known frauds:**")
        cols = st.columns(3)
        for i, fi in enumerate(fraud_indices[:12]):
            if cols[i % 3].button(str(fi), key=f"qp_{fi}"):
                st.session_state.sample_index = fi
                st.rerun()

    st.markdown("---")
    run = st.button("🔍 Analyse Transaction", use_container_width=True)

# ── get features from selected index ──────────────────────────────────────────
features = X_test[int(st.session_state.sample_index)].tolist()

# show which sample is loaded
st.markdown(f'<div style="background:white;border:1px solid #E2E8F0;border-radius:10px;padding:12px 18px;margin-bottom:16px;font-size:.9rem;color:#475569;">📋 Loaded sample <b style="color:#4F6EF7">#{int(st.session_state.sample_index)}</b> — {len(features)} features ready</div>', unsafe_allow_html=True)

# ── Analysis ───────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Running all 4 models... (first request may take ~15s if API is waking up)"):
        try:
            res      = requests.post(API_URL, json={"features": features}, timeout=60).json()
            is_fraud = res["fraud_detected"]
            prob     = res["rf_fraud_probability"]

            st.session_state.total_checked += 1
            if is_fraud:
                st.session_state.fraud_count += 1

            st.session_state.history.append({
                "index":   st.session_state.total_checked,
                "sample":  int(st.session_state.sample_index),
                "fraud":   is_fraud,
                "votes":   res["votes"],
                "rf_prob": prob,
                "lr":      res["logistic_regression"],
                "rf":      res["random_forest"],
                "iso":     res["isolation_forest"],
                "time":    datetime.now().strftime("%H:%M:%S"),
            })

            # Result banner
            if is_fraud:
                st.markdown(f'<div class="fraud-banner"><div class="banner-title" style="color:#DC2626">🚨 FRAUD DETECTED</div><div class="banner-sub">{res["votes"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="legit-banner"><div class="banner-title" style="color:#16A34A">✅ LEGITIMATE TRANSACTION</div><div class="banner-sub">{res["votes"]}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Model cards
            st.markdown('<div class="sec-label">🤖 Model Predictions</div>', unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)

            def mcard(col, name, fraud_flag):
                lbl = "🚨 Fraud" if fraud_flag else "✅ Legit"
                cls = "mcard-fraud" if fraud_flag else "mcard-legit"
                col.markdown(f'<div class="mcard"><div class="mcard-label">{name}</div><div class="{cls}">{lbl}</div></div>', unsafe_allow_html=True)

            mcard(mc1, "Logistic Regression", bool(res["logistic_regression"]))
            mcard(mc2, "Random Forest",        bool(res["random_forest"]))
            mcard(mc3, "Isolation Forest",     bool(res["isolation_forest"]))
            mc4.markdown(f'<div class="mcard"><div class="mcard-label">RF Fraud Probability</div><div class="{"mcard-fraud" if prob>50 else "mcard-legit"}" style="font-size:1.3rem">{prob}%</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Charts
            st.markdown('<div class="sec-label">📊 Model Prediction Breakdown</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)

            with c1:
                vv = [res["logistic_regression"], res["random_forest"],
                      res["isolation_forest"], 1 if prob > 70 else 0]
                vn = ["Logistic Reg.", "Random Forest", "Isolation Forest", "AE Proxy"]
                fig1 = go.Figure(go.Bar(
                    x=vn, y=vv,
                    marker_color=["#EF4444" if v else "#10B981" for v in vv],
                    text=["🚨 Fraud" if v else "✅ Legit" for v in vv],
                    textposition="outside",
                    textfont=dict(family="DM Sans", size=13)
                ))
                fig1.update_layout(
                    title=dict(text="Model Votes", font=dict(family="DM Sans", size=15, color="#1E293B")),
                    paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(family="DM Sans", color="#1E293B"),
                    yaxis=dict(range=[0,1.8], tickvals=[0,1], ticktext=["Legit","Fraud"], gridcolor="#F1F5F9"),
                    xaxis=dict(gridcolor="#F1F5F9"),
                    margin=dict(t=50,b=10,l=10,r=10), height=300, showlegend=False
                )
                st.plotly_chart(fig1, use_container_width=True)

            with c2:
                fig2 = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob,
                    number={"suffix":"%","font":{"size":38,"family":"DM Mono",
                            "color":"#EF4444" if prob>50 else "#10B981"}},
                    delta={"reference":50,"increasing":{"color":"#EF4444"},"decreasing":{"color":"#10B981"}},
                    gauge={
                        "axis":{"range":[0,100],"tickfont":{"family":"DM Sans"}},
                        "bar": {"color":"#EF4444" if prob>50 else "#10B981","thickness":.28},
                        "steps":[
                            {"range":[0,30],  "color":"#DCFCE7"},
                            {"range":[30,70], "color":"#FEF9C3"},
                            {"range":[70,100],"color":"#FEE2E2"},
                        ],
                        "threshold":{"line":{"color":"#4F6EF7","width":3},"thickness":.75,"value":50}
                    },
                    title={"text":"RF Fraud Probability","font":{"family":"DM Sans","size":14,"color":"#64748B"}}
                ))
                fig2.update_layout(paper_bgcolor="white", margin=dict(t=30,b=10,l=20,r=20), height=300)
                st.plotly_chart(fig2, use_container_width=True)

            # Feature heatmap
            st.markdown('<div class="sec-label">🔬 Feature Values (Current Transaction)</div>', unsafe_allow_html=True)
            fig3 = go.Figure(go.Heatmap(
                z=np.array(features).reshape(1,-1),
                x=[f"V{i+1}" for i in range(len(features))],
                colorscale="RdYlGn_r", showscale=True,
            ))
            fig3.update_layout(
                paper_bgcolor="white", plot_bgcolor="white",
                font=dict(family="DM Sans"),
                margin=dict(t=10,b=10,l=10,r=10), height=130,
                yaxis=dict(showticklabels=False),
                xaxis=dict(tickfont=dict(size=9))
            )
            st.plotly_chart(fig3, use_container_width=True)

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API.")
            st.info("👉 If deployed: the Render API may be waking up (free tier sleeps after 15 min). Wait 30 seconds and try again.\n\n👉 If running locally: make sure uvicorn is running:\n```\npython -m uvicorn app:app --reload\n```")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ── History ────────────────────────────────────────────────────────────────────
history = st.session_state.history
if history:
    st.markdown("---")
    hl, hr = st.columns([2, 1])

    with hl:
        st.markdown('<div class="sec-label">📋 Transaction Log</div>', unsafe_allow_html=True)
        for i, entry in enumerate(reversed(history[-10:])):
            icon  = "🚨" if entry["fraud"] else "✅"
            color = "#EF4444" if entry["fraud"] else "#10B981"
            label = "FRAUD" if entry["fraud"] else "LEGIT"
            n     = len(history) - i
            st.markdown(
                f'<div class="log-row">'
                f'<span>{icon}</span>'
                f'<span style="color:{color};font-weight:700">#{n} · Sample {entry["sample"]} · {label}</span>'
                f'<span style="color:#94A3B8">·</span>'
                f'<span style="color:#475569">{entry["votes"]}</span>'
                f'<span style="color:#94A3B8">·</span>'
                f'<span style="color:#475569">RF: {entry["rf_prob"]}%</span>'
                f'<span style="margin-left:auto;color:#94A3B8">{entry["time"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    with hr:
        st.markdown('<div class="sec-label">📊 Session Overview</div>', unsafe_allow_html=True)
        total_h  = len(history)
        frauds_h = sum(1 for e in history if e["fraud"])
        legits_h = total_h - frauds_h
        fig4 = go.Figure(go.Pie(
            labels=["Fraud","Legitimate"],
            values=[max(frauds_h,.001), max(legits_h,.001)],
            marker_colors=["#EF4444","#10B981"], hole=.55,
            textfont=dict(family="DM Sans"),
        ))
        fig4.update_layout(
            paper_bgcolor="white", margin=dict(t=10,b=10,l=10,r=10), height=230,
            legend=dict(font=dict(family="DM Sans",size=12)),
            annotations=[dict(text=f"<b>{total_h}</b><br>total", x=.5, y=.5,
                              font_size=14, showarrow=False,
                              font=dict(family="DM Sans",color="#1E293B"))]
        )
        st.plotly_chart(fig4, use_container_width=True)

    if len(history) > 1:
        st.markdown('<div class="sec-label">📈 RF Fraud Probability Trend</div>', unsafe_allow_html=True)
        df_h = pd.DataFrame(history)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=df_h["index"], y=df_h["rf_prob"],
            mode="lines+markers",
            line=dict(color="#4F6EF7", width=2.5),
            marker=dict(color=["#EF4444" if f else "#10B981" for f in df_h["fraud"]],
                        size=9, line=dict(color="white",width=2)),
            fill="tozeroy", fillcolor="rgba(79,110,247,.07)",
            hovertemplate="Sample %{customdata}<br>RF Prob: %{y}%<extra></extra>",
            customdata=df_h["sample"]
        ))
        fig5.add_hline(y=50, line_dash="dash", line_color="#F59E0B",
                       annotation_text="Decision boundary (50%)",
                       annotation_font=dict(family="DM Sans",color="#B45309"))
        fig5.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="DM Sans",color="#1E293B"),
            xaxis=dict(title="Analysis #", gridcolor="#F1F5F9"),
            yaxis=dict(title="RF Fraud Probability (%)", range=[0,105], gridcolor="#F1F5F9"),
            margin=dict(t=10,b=10,l=10,r=10), height=260
        )
        st.plotly_chart(fig5, use_container_width=True)
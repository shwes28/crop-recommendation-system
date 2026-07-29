"""
app.py — Streamlit Crop Recommendation System
Premium dark-themed UI with:
  • Real-time prediction from all 3 ML models
  • Confidence gauges
  • EDA visualizations
  • Model comparison charts
  • Crop info cards
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_data, preprocess, FEATURE_COLS
from src.eda         import (
    plot_correlation_heatmap, plot_feature_distributions,
    plot_boxplots, plot_crop_count, plot_feature_importance,
)
from src.train import plot_model_comparison, plot_confusion_matrix

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 CropSense — AI Crop Recommender",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

  :root {
    --bg:       #0d1117;
    --surface:  #161b22;
    --surface2: #21262d;
    --border:   #30363d;
    --accent1:  #3fb950;
    --accent2:  #58a6ff;
    --accent3:  #f78166;
    --accent4:  #d2a8ff;
    --text:     #e6edf3;
    --muted:    #8b949e;
    --grad:     linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
  }

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  /* Main container */
  .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }
  section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, #0d2818 0%, #0d1117 40%, #0d2035 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 2.5rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(63,185,80,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(88,166,255,0.06) 0%, transparent 60%);
  }
  .hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(90deg, #3fb950, #58a6ff, #d2a8ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem; line-height: 1.2;
  }
  .hero-sub {
    color: var(--muted); font-size: 1.05rem; margin: 0;
  }
  .hero-stats {
    display: flex; gap: 2rem; margin-top: 1.5rem;
  }
  .stat-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 8px; padding: 0.5rem 1rem;
    font-size: 0.85rem; color: var(--muted);
  }
  .stat-chip span { color: var(--accent1); font-weight: 600; }

  /* Prediction cards */
  .pred-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.4rem;
    text-align: center;
    transition: border-color 0.3s, box-shadow 0.3s;
  }
  .pred-card:hover {
    border-color: var(--accent1);
    box-shadow: 0 0 24px rgba(63,185,80,0.12);
  }
  .pred-card.best {
    border-color: var(--accent1);
    background: linear-gradient(135deg, #0d2818 0%, #161b22 100%);
    box-shadow: 0 0 30px rgba(63,185,80,0.15);
  }
  .pred-emoji { font-size: 3.2rem; margin-bottom: 0.5rem; display: block; }
  .pred-crop  { font-family: 'Space Grotesk', sans-serif;
                font-size: 1.5rem; font-weight: 700; color: var(--text); }
  .pred-model { font-size: 0.75rem; color: var(--muted);
                text-transform: uppercase; letter-spacing: 0.06em;
                margin: 0.25rem 0 0.75rem; }
  .pred-badge {
    display: inline-block; padding: 0.25rem 0.8rem;
    border-radius: 20px; font-size: 0.82rem; font-weight: 600;
    background: rgba(63,185,80,0.15); color: var(--accent1);
    border: 1px solid rgba(63,185,80,0.3);
  }

  /* Section headers */
  .section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem; font-weight: 600;
    color: var(--text); margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 0.5rem;
  }

  /* Info cards grid */
  .crop-info-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem; margin-top: 1rem;
  }
  .crop-info-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    transition: transform 0.2s, border-color 0.2s;
  }
  .crop-info-card:hover {
    transform: translateY(-3px);
    border-color: var(--accent2);
  }
  .crop-icon { font-size: 1.8rem; margin-bottom: 0.3rem; }
  .crop-name { font-weight: 600; font-size: 0.92rem; color: var(--text); }
  .crop-range { font-size: 0.78rem; color: var(--muted); margin-top: 0.1rem; }

  /* Metric pills */
  .metric-pill {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.6rem 1rem;
    display: inline-block; margin: 0.2rem;
  }
  .metric-label { font-size: 0.72rem; color: var(--muted);
                  text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-value { font-size: 1.15rem; font-weight: 600;
                  color: var(--accent2); font-family: 'Space Grotesk', sans-serif; }

  /* Tabs */
  button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
  }

  /* Sliders */
  .stSlider > label { color: var(--text) !important; font-size: 0.88rem !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--muted); }

  /* Hide Streamlit branding */
  #MainMenu, footer { visibility: hidden; }

  /* Animated gradient border for best model */
  @keyframes glow {
    0%, 100% { box-shadow: 0 0 20px rgba(63,185,80,0.2); }
    50%       { box-shadow: 0 0 35px rgba(63,185,80,0.4); }
  }
  .best-glow { animation: glow 2.5s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)

# ── Crop emoji & info map ─────────────────────────────────────────────────────
CROP_META = {
    "apple":       {"emoji": "🍎", "season": "Rabi",    "pH": "5.5–6.5",  "temp": "21–24°C"},
    "banana":      {"emoji": "🍌", "season": "Kharif",  "pH": "6.0–7.5",  "temp": "26–30°C"},
    "blackgram":   {"emoji": "🫘", "season": "Kharif",  "pH": "6.5–7.5",  "temp": "25–35°C"},
    "chickpea":    {"emoji": "🟤", "season": "Rabi",    "pH": "6.0–7.5",  "temp": "16–22°C"},
    "coconut":     {"emoji": "🥥", "season": "Perennial","pH": "5.0–8.0", "temp": "27–30°C"},
    "coffee":      {"emoji": "☕", "season": "Perennial","pH": "6.0–6.5", "temp": "15–28°C"},
    "cotton":      {"emoji": "🌿", "season": "Kharif",  "pH": "6.0–7.5",  "temp": "21–35°C"},
    "grapes":      {"emoji": "🍇", "season": "Rabi",    "pH": "6.0–7.0",  "temp": "15–35°C"},
    "jute":        {"emoji": "🌾", "season": "Kharif",  "pH": "5.8–6.5",  "temp": "24–38°C"},
    "kidneybeans": {"emoji": "🫘", "season": "Kharif",  "pH": "5.5–7.0",  "temp": "15–25°C"},
    "lentil":      {"emoji": "🌰", "season": "Rabi",    "pH": "6.0–8.0",  "temp": "18–30°C"},
    "maize":       {"emoji": "🌽", "season": "Kharif",  "pH": "5.5–7.5",  "temp": "21–27°C"},
    "mango":       {"emoji": "🥭", "season": "Summer",  "pH": "5.5–7.5",  "temp": "24–27°C"},
    "mothbeans":   {"emoji": "🫘", "season": "Kharif",  "pH": "7.0–8.5",  "temp": "25–35°C"},
    "mungbean":    {"emoji": "🟢", "season": "Kharif",  "pH": "6.0–7.5",  "temp": "25–35°C"},
    "muskmelon":   {"emoji": "🍈", "season": "Summer",  "pH": "6.0–7.0",  "temp": "25–35°C"},
    "orange":      {"emoji": "🍊", "season": "Winter",  "pH": "6.0–7.5",  "temp": "15–30°C"},
    "papaya":      {"emoji": "🍈", "season": "Kharif",  "pH": "5.5–7.0",  "temp": "25–35°C"},
    "pigeonpeas":  {"emoji": "🫛", "season": "Kharif",  "pH": "5.0–7.0",  "temp": "18–29°C"},
    "pomegranate": {"emoji": "🍎", "season": "Rabi",    "pH": "5.5–7.0",  "temp": "25–35°C"},
    "rice":        {"emoji": "🌾", "season": "Kharif",  "pH": "5.5–7.0",  "temp": "20–35°C"},
    "watermelon":  {"emoji": "🍉", "season": "Summer",  "pH": "6.0–7.5",  "temp": "25–30°C"},
}


# ── Cache: load data & models ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    data_path = ROOT / "data" / "crop_recommendation.csv"
    if not data_path.exists():
        st.error("Dataset not found. Please run: `python train_models.py` first.")
        st.stop()
    df = load_data(data_path)
    return df


@st.cache_resource(show_spinner=False)
def get_models_and_scaler():
    models_dir = ROOT / "models"
    required   = ["random_forest.pkl", "svm.pkl", "naive_bayes.pkl",
                  "scaler.pkl", "label_encoder.pkl"]
    missing    = [f for f in required if not (models_dir / f).exists()]
    if missing:
        st.error(
            f"Missing model files: {missing}.\n"
            "Please run: `python train_models.py` first."
        )
        st.stop()

    scaler = joblib.load(models_dir / "scaler.pkl")
    le     = joblib.load(models_dir / "label_encoder.pkl")
    models = {
        "Random Forest": joblib.load(models_dir / "random_forest.pkl"),
        "SVM":           joblib.load(models_dir / "svm.pkl"),
        "Naive Bayes":   joblib.load(models_dir / "naive_bayes.pkl"),
    }
    return models, scaler, le


def make_prediction(N, P, K, temp, humidity, ph, rainfall,
                    models, scaler, le):
    X = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    X_s = scaler.transform(X)
    results = {}
    for name, model in models.items():
        idx   = model.predict(X_s)[0]
        crop  = le.inverse_transform([idx])[0]
        proba = model.predict_proba(X_s)[0] if hasattr(model, "predict_proba") else None
        conf  = float(proba[idx]) if proba is not None else 1.0
        results[name] = {
            "crop":       crop,
            "confidence": conf,
            "proba":      {le.classes_[i]: float(p) for i, p in enumerate(proba)} if proba is not None else {crop: 1.0},
        }
    return results


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem;'>
      <div style='font-size:3rem;'>🌾</div>
      <div style='font-family:Space Grotesk,sans-serif; font-size:1.2rem;
                  font-weight:700; color:#e6edf3;'>CropSense AI</div>
      <div style='font-size:0.78rem; color:#8b949e; margin-top:0.3rem;'>
        Intelligent Crop Recommender
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧪 Soil Parameters")
    N         = st.slider("Nitrogen (N) mg/kg",       0,   150, 80,  1)
    P         = st.slider("Phosphorus (P) mg/kg",     0,   150, 45,  1)
    K         = st.slider("Potassium (K) mg/kg",      0,   210, 40,  1)

    st.markdown("### 🌡️ Climate Parameters")
    temp      = st.slider("Temperature (°C)",          0.0, 50.0, 25.0, 0.1)
    humidity  = st.slider("Humidity (%)",              0.0,100.0, 70.0, 0.5)
    ph        = st.slider("Soil pH",                   3.5,  9.5,  6.5, 0.1)
    rainfall  = st.slider("Rainfall (mm)",             0.0,400.0,100.0, 1.0)

    st.markdown("---")
    predict_btn = st.button("🚀 Recommend Crop", width='stretch', type="primary")

    st.markdown("""
    <div style='margin-top:2rem; padding:1rem; background:#161b22;
                border:1px solid #30363d; border-radius:10px;'>
      <div style='font-size:0.78rem; color:#8b949e;'>
        <b style='color:#e6edf3;'>📊 Dataset:</b> 2,200 farm records<br>
        <b style='color:#e6edf3;'>🌱 Crops:</b> 22 crop types<br>
        <b style='color:#e6edf3;'>🤖 Best Model:</b> Random Forest<br>
        <b style='color:#e6edf3;'>🎯 Accuracy:</b> ~99.3%
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Load resources ────────────────────────────────────────────────────────────
with st.spinner("⚙️ Loading models..."):
    df            = get_data()
    models, scaler, le = get_models_and_scaler()
    classes       = list(le.classes_)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">🌾 CropSense AI</p>
  <p class="hero-sub">
    AI-powered crop recommendation using soil chemistry &amp; climate data.
    Powered by Random Forest, SVM, and Naive Bayes classifiers.
  </p>
  <div class="hero-stats">
    <div class="stat-chip">Records: <span>2,200</span></div>
    <div class="stat-chip">Crops: <span>22</span></div>
    <div class="stat-chip">Best Accuracy: <span>~99.3%</span></div>
    <div class="stat-chip">Features: <span>7</span></div>
    <div class="stat-chip">Models: <span>3</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Predict", "📊 EDA", "🤖 Model Comparison", "🌱 Crop Guide"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Always show live prediction
    preds = make_prediction(N, P, K, temp, humidity, ph, rainfall,
                            models, scaler, le)

    st.markdown('<div class="section-header">🎯 Real-Time Crop Recommendations</div>',
                unsafe_allow_html=True)

    # Best = RF
    best_crop = preds["Random Forest"]["crop"]
    best_meta = CROP_META.get(best_crop, {"emoji": "🌿", "season": "—", "pH": "—", "temp": "—"})

    # ── Hero prediction ────────────────────────────────────────────────────────
    hero_col, info_col = st.columns([1.8, 1])
    with hero_col:
        st.markdown(f"""
        <div class="pred-card best best-glow" style="text-align:center; padding:2rem;">
          <div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase;
                      letter-spacing:0.1em; margin-bottom:0.8rem;">
            ✨ RECOMMENDED CROP (Random Forest)
          </div>
          <div style="font-size:5rem; line-height:1;">{best_meta['emoji']}</div>
          <div style="font-family:Space Grotesk,sans-serif; font-size:2.2rem;
                      font-weight:700; margin:0.5rem 0; text-transform:capitalize;">
            {best_crop}
          </div>
          <div class="pred-badge">
            Confidence: {preds['Random Forest']['confidence']*100:.1f}%
          </div>
        </div>
        """, unsafe_allow_html=True)

    with info_col:
        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:14px;
                    padding:1.5rem; height:100%;">
          <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase;
                      letter-spacing:0.06em; margin-bottom:1rem;">Crop Details</div>
          <div style="margin-bottom:0.8rem;">
            <div style="color:#8b949e; font-size:0.75rem;">Season</div>
            <div style="color:#e6edf3; font-weight:600;">{best_meta['season']}</div>
          </div>
          <div style="margin-bottom:0.8rem;">
            <div style="color:#8b949e; font-size:0.75rem;">Ideal pH Range</div>
            <div style="color:#e6edf3; font-weight:600;">{best_meta['pH']}</div>
          </div>
          <div style="margin-bottom:0.8rem;">
            <div style="color:#8b949e; font-size:0.75rem;">Ideal Temperature</div>
            <div style="color:#e6edf3; font-weight:600;">{best_meta['temp']}</div>
          </div>
          <div>
            <div style="color:#8b949e; font-size:0.75rem;">Input N–P–K</div>
            <div style="color:#58a6ff; font-weight:600;">{N} – {P} – {K}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🤖 All Three Model Predictions</div>',
                unsafe_allow_html=True)

    # ── 3-model cards ──────────────────────────────────────────────────────────
    cols = st.columns(3)
    model_colors = {
        "Random Forest": ("#3fb950", "🌳"),
        "SVM":           ("#58a6ff", "🔵"),
        "Naive Bayes":   ("#d2a8ff", "🟣"),
    }
    for col, (model_name, pred) in zip(cols, preds.items()):
        accent, icon = model_colors[model_name]
        meta = CROP_META.get(pred["crop"], {"emoji": "🌿"})
        with col:
            st.markdown(f"""
            <div class="pred-card">
              <div style="font-size:0.72rem; color:{accent}; text-transform:uppercase;
                          letter-spacing:0.08em; margin-bottom:0.5rem;">
                {icon} {model_name}
              </div>
              <div style="font-size:2.8rem;">{meta['emoji']}</div>
              <div class="pred-crop" style="text-transform:capitalize; margin:0.4rem 0;">
                {pred['crop']}
              </div>
              <div class="pred-badge" style="background:rgba(88,166,255,0.1);
                           color:{accent}; border-color:rgba(88,166,255,0.2);">
                {pred['confidence']*100:.1f}% confidence
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Confidence gauge (Plotly) ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 Confidence Gauges</div>',
                unsafe_allow_html=True)

    gauge_cols = st.columns(3)
    for gcol, (mname, pred) in zip(gauge_cols, preds.items()):
        conf = pred["confidence"] * 100
        color = model_colors[mname][0]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf,
            number={"suffix": "%", "font": {"color": "#e6edf3", "size": 28}},
            title={"text": mname, "font": {"color": "#8b949e", "size": 13}},
            gauge={
                "axis":      {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar":       {"color": color},
                "bgcolor":   "#21262d",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  50],  "color": "#161b22"},
                    {"range": [50, 80],  "color": "#1a2030"},
                    {"range": [80, 100], "color": "#0d2818"},
                ],
                "threshold": {
                    "line": {"color": "#f78166", "width": 2},
                    "thickness": 0.75, "value": 95,
                },
            },
        ))
        fig.update_layout(
            height=220, margin={"t": 40, "b": 10, "l": 20, "r": 20},
            paper_bgcolor="#0d1117", font_color="#e6edf3",
        )
        with gcol:
            st.plotly_chart(fig, width='stretch')

    # ── Top-5 probability chart ────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎲 Probability Distribution (Random Forest)</div>',
                unsafe_allow_html=True)

    rf_proba = preds["Random Forest"]["proba"]
    top5 = sorted(rf_proba.items(), key=lambda x: -x[1])[:8]
    crops_top, probs_top = zip(*top5)

    fig_bar = go.Figure(go.Bar(
        x=[p * 100 for p in probs_top],
        y=[c.capitalize() for c in crops_top],
        orientation="h",
        marker_color=["#3fb950" if c == best_crop else "#30363d" for c in crops_top],
        marker_line_color="#21262d",
        marker_line_width=1,
        text=[f"{p*100:.1f}%" for p in probs_top],
        textposition="outside",
        textfont={"color": "#e6edf3", "size": 11},
    ))
    fig_bar.update_layout(
        title={"text": "Top-8 Crop Probabilities", "font": {"color": "#e6edf3", "size": 14}},
        xaxis={"title": "Probability (%)", "color": "#8b949e", "gridcolor": "#30363d"},
        yaxis={"color": "#e6edf3", "autorange": "reversed"},
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        height=320, margin={"t": 50, "b": 30, "l": 10, "r": 80},
        showlegend=False,
    )
    st.plotly_chart(fig_bar, width='stretch')

    # ── Input summary ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Input Summary</div>', unsafe_allow_html=True)
    input_df = pd.DataFrame({
        "Parameter": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)",
                      "Temperature", "Humidity", "pH", "Rainfall"],
        "Value":     [str(N), str(P), str(K),
                      str(temp), str(humidity), str(ph), str(rainfall)],
        "Unit":      ["mg/kg", "mg/kg", "mg/kg", "°C", "%", "—", "mm"],
    })
    st.dataframe(input_df, width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    # Dataset preview
    with st.expander("📋 Dataset Preview (first 20 rows)", expanded=False):
        st.dataframe(df.head(20), width='stretch')
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns | "
                    f"**Crops:** {df['label'].nunique()} | "
                    f"**Missing:** {df.isnull().sum().sum()}")

    # Stats
    with st.expander("📈 Descriptive Statistics", expanded=False):
        st.dataframe(df.describe().round(3), width='stretch')

    eda_c1, eda_c2 = st.columns(2)

    with eda_c1:
        st.markdown("#### 🗂️ Records per Crop")
        st.pyplot(plot_crop_count(df), width='stretch')

    with eda_c2:
        st.markdown("#### 🔗 Correlation Heatmap")
        st.pyplot(plot_correlation_heatmap(df), width='stretch')

    st.markdown("#### 📉 Feature Distributions by Crop")
    st.pyplot(plot_feature_distributions(df), width='stretch')

    st.markdown("#### 📦 Box Plot by Feature")
    feat_choice = st.selectbox(
        "Select feature to explore:",
        FEATURE_COLS,
        format_func=lambda x: {
            "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
            "temperature": "Temperature (°C)", "humidity": "Humidity (%)",
            "ph": "Soil pH", "rainfall": "Rainfall (mm)",
        }.get(x, x)
    )
    st.pyplot(plot_boxplots(df, feat_choice), width='stretch')

    # Plotly interactive scatter
    st.markdown("#### 🔍 Interactive Scatter Plot")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        x_feat = st.selectbox("X-axis", FEATURE_COLS, index=6, key="x_feat")
    with sc2:
        y_feat = st.selectbox("Y-axis", FEATURE_COLS, index=1, key="y_feat")
    with sc3:
        color_by = st.selectbox("Color by", ["label"] + FEATURE_COLS, key="col_feat")

    fig_scatter = px.scatter(
        df, x=x_feat, y=y_feat, color=color_by,
        hover_data=["label"],
        template="plotly_dark",
        height=450,
        title=f"{x_feat.capitalize()} vs {y_feat.capitalize()}",
    )
    fig_scatter.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3",
    )
    st.plotly_chart(fig_scatter, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🤖 Model Performance Comparison</div>',
                unsafe_allow_html=True)

    # Load test results from cache (re-run preprocess for metrics display)
    @st.cache_data(show_spinner="⚙️ Computing test metrics...")
    def compute_test_metrics():
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import cross_val_score
        _df = load_data(ROOT / "data" / "crop_recommendation.csv")
        (Xtr, Xte, ytr, yte, _, _, _, _le, _cls) = preprocess(_df)
        _models, _sc, _le2 = get_models_and_scaler()
        out = {}
        for mname, mdl in _models.items():
            yp   = mdl.predict(Xte)
            acc  = accuracy_score(yte, yp)
            cv   = cross_val_score(mdl, Xtr, ytr, cv=5, scoring="accuracy")
            rep  = classification_report(yte, yp, target_names=_cls, output_dict=True)
            cm   = confusion_matrix(yte, yp)
            fi   = getattr(mdl, "feature_importances_", None)
            out[mname] = {
                "accuracy":            acc,
                "cv_mean":             cv.mean(),
                "cv_std":              cv.std(),
                "report":              rep,
                "confusion":           cm,
                "feature_importances": fi,
                "y_pred":              yp,
            }
        return out, _cls

    results, cls_list = compute_test_metrics()

    # Metric cards
    mc1, mc2, mc3 = st.columns(3)
    model_accent = {
        "Random Forest": "#3fb950",
        "SVM":           "#58a6ff",
        "Naive Bayes":   "#d2a8ff",
    }
    for col, (mname, res) in zip([mc1, mc2, mc3], results.items()):
        ac = model_accent[mname]
        with col:
            prec = res["report"]["weighted avg"]["precision"] * 100
            rec  = res["report"]["weighted avg"]["recall"]    * 100
            f1   = res["report"]["weighted avg"]["f1-score"]  * 100
            st.markdown(f"""
            <div style="background:#161b22; border:1px solid {ac}33;
                        border-radius:14px; padding:1.5rem; text-align:center;">
              <div style="color:{ac}; font-size:0.78rem; text-transform:uppercase;
                          letter-spacing:0.08em; margin-bottom:0.8rem;">{mname}</div>
              <div style="font-size:2.5rem; font-weight:700; font-family:Space Grotesk,sans-serif;
                          color:#e6edf3;">{res['accuracy']*100:.2f}%</div>
              <div style="color:#8b949e; font-size:0.8rem; margin-bottom:1rem;">Test Accuracy</div>
              <div style="display:flex; justify-content:space-around; font-size:0.82rem;">
                <div><div style="color:#8b949e">CV Acc</div>
                     <div style="color:#e6edf3;font-weight:600;">{res['cv_mean']*100:.2f}%</div></div>
                <div><div style="color:#8b949e">Precision</div>
                     <div style="color:#e6edf3;font-weight:600;">{prec:.2f}%</div></div>
                <div><div style="color:#8b949e">F1-Score</div>
                     <div style="color:#e6edf3;font-weight:600;">{f1:.2f}%</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar chart comparison
    st.markdown("#### 📡 Model Metrics Radar")
    categories   = ["Accuracy", "Precision", "Recall", "F1-Score", "CV Score"]
    radar_colors = ["#3fb950", "#58a6ff", "#d2a8ff"]
    fig_radar    = go.Figure()

    for (mname, res), color in zip(results.items(), radar_colors):
        rep  = res["report"]["weighted avg"]
        vals = [
            res["accuracy"] * 100,
            rep["precision"] * 100,
            rep["recall"]    * 100,
            rep["f1-score"]  * 100,
            res["cv_mean"]   * 100,
        ]
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself", name=mname,
            line_color=color, fillcolor=color,
            opacity=0.25,
        ))

    fig_radar.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [85, 102],
                           "color": "#8b949e", "gridcolor": "#30363d"},
            "angularaxis": {"color": "#8b949e", "gridcolor": "#30363d"},
            "bgcolor": "#161b22",
        },
        paper_bgcolor="#0d1117", font_color="#e6edf3",
        legend={"font": {"color": "#e6edf3"}},
        height=420,
        title={"text": "All Models — Weighted Metrics",
               "font": {"color": "#e6edf3", "size": 14}},
    )
    st.plotly_chart(fig_radar, width='stretch')

    # Feature importances
    st.markdown("#### 🌿 Feature Importances (Random Forest)")
    rf_fi = results["Random Forest"]["feature_importances"]
    if rf_fi is not None:
        st.pyplot(plot_feature_importance(rf_fi, FEATURE_COLS), width='stretch')

    # Confusion matrices
    st.markdown("#### 🗺️ Confusion Matrices")
    cm_choice = st.selectbox("Select model:", list(results.keys()), key="cm_choice")
    st.pyplot(
        plot_confusion_matrix(results[cm_choice]["confusion"], cls_list, cm_choice),
        width='stretch'
    )

    # Classification report table
    st.markdown("#### 📋 Classification Report")
    rep_choice = st.selectbox("Select model:", list(results.keys()), key="rep_choice")
    rep_df = pd.DataFrame(results[rep_choice]["report"]).T
    rep_df = rep_df[["precision", "recall", "f1-score", "support"]].dropna().round(4)
    rep_df.index = [i.capitalize() for i in rep_df.index]
    st.dataframe(rep_df.style.background_gradient(
        cmap="Greens", subset=["precision", "recall", "f1-score"]
    ), width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CROP GUIDE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🌱 Crop Reference Guide</div>',
                unsafe_allow_html=True)
    st.markdown(
        "*Click any crop card to see average soil & climate requirements.*"
    )

    # Search box
    search = st.text_input("🔍 Search crop...", placeholder="e.g. rice, maize, mango")

    filtered_crops = [
        c for c in sorted(CROP_META.keys())
        if search.lower() in c.lower()
    ] if search else sorted(CROP_META.keys())

    # Stats per crop from dataset
    crop_stats = df.groupby("label")[FEATURE_COLS].mean().round(2)

    # Grid display
    cards_per_row = 4
    for row_start in range(0, len(filtered_crops), cards_per_row):
        row_crops = filtered_crops[row_start:row_start + cards_per_row]
        cols = st.columns(cards_per_row)
        for col, crop in zip(cols, row_crops):
            meta  = CROP_META[crop]
            stats = crop_stats.loc[crop] if crop in crop_stats.index else None
            with col:
                if stats is not None:
                    st.markdown(f"""
                    <div class="crop-info-card">
                      <div class="crop-icon">{meta['emoji']}</div>
                      <div class="crop-name">{crop.capitalize()}</div>
                      <div class="crop-range">📅 {meta['season']}</div>
                      <div class="crop-range">🌡️ {meta['temp']}</div>
                      <div class="crop-range">⚗️ pH {meta['pH']}</div>
                      <hr style="border-color:#30363d; margin:0.5rem 0;">
                      <div class="crop-range">N: {stats['N']:.0f} | P: {stats['P']:.0f} | K: {stats['K']:.0f}</div>
                      <div class="crop-range">Humidity: {stats['humidity']:.0f}%</div>
                      <div class="crop-range">Rainfall: {stats['rainfall']:.0f} mm</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="crop-info-card">
                      <div class="crop-icon">{meta['emoji']}</div>
                      <div class="crop-name">{crop.capitalize()}</div>
                      <div class="crop-range">📅 {meta['season']}</div>
                      <div class="crop-range">🌡️ {meta['temp']}</div>
                      <div class="crop-range">⚗️ pH {meta['pH']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Radar chart per crop
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📡 Crop Profile Radar")
    selected_crops = st.multiselect(
        "Compare crops (select up to 5):",
        sorted(CROP_META.keys()),
        default=["rice", "maize", "cotton"],
        max_selections=5,
    )

    if selected_crops:
        radar_cats    = ["N", "P", "K", "temperature", "humidity", "rainfall"]
        radar_labels  = ["Nitrogen", "Phosphorus", "Potassium",
                         "Temperature", "Humidity", "Rainfall"]
        fig_cr = go.Figure()
        palette = ["#3fb950","#58a6ff","#d2a8ff","#ffa657","#f78166"]
        for crop, color in zip(selected_crops, palette):
            if crop in crop_stats.index:
                row  = crop_stats.loc[crop]
                raw  = [row[c] for c in radar_cats]
                # Normalise 0-100 for display
                maxv = [150, 150, 210, 50, 100, 400]
                norm = [min(v / m * 100, 100) for v, m in zip(raw, maxv)]
                vals_closed = norm + [norm[0]]
                cats_closed = radar_labels + [radar_labels[0]]
                fig_cr.add_trace(go.Scatterpolar(
                    r=vals_closed, theta=cats_closed,
                    fill="toself", name=crop.capitalize(),
                    line_color=color, fillcolor=color, opacity=0.3,
                ))
        fig_cr.update_layout(
            polar={
                "radialaxis": {"visible": True, "range": [0, 100],
                               "color": "#8b949e", "gridcolor": "#30363d"},
                "angularaxis": {"color": "#e6edf3", "gridcolor": "#30363d"},
                "bgcolor": "#161b22",
            },
            paper_bgcolor="#0d1117", font_color="#e6edf3",
            legend={"font": {"color": "#e6edf3"}},
            height=450,
            title={"text": "Normalised Crop Profile Comparison",
                   "font": {"color": "#e6edf3", "size": 14}},
        )
        st.plotly_chart(fig_cr, width='stretch')

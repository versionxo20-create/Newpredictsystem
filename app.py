import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Hypertension Risk Prediction System",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# Custom CSS for cleaner look
# -------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .hypertension {
        background-color: #ffebee;
        border: 2px solid #e53935;
    }
    .non-hypertension {
        background-color: #e8f5e9;
        border: 2px solid #43a047;
    }
    .disclaimer-box {
        background-color: #fff3e0;
        border-left: 6px solid #ef6c00;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 1.2rem 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .footer {
        text-align: center;
        color: #777;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load model (cached)
# -------------------------------------------------
MODEL_PATH = Path(__file__).parent / "hypertension_random_forest.pkl"

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_model()

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<p class="main-title">❤️ Hypertension Risk Prediction</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Random Forest model trained on NHANES 2017–2018 data (7 predictors)</p>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# PROMINENT DISCLAIMER (top)
# -------------------------------------------------
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ IMPORTANT MEDICAL DISCLAIMER</strong><br><br>
    This is a <strong>research / educational tool only</strong>. 
    It does <strong>NOT</strong> provide a medical diagnosis and must never be used as a substitute 
    for professional medical advice, diagnosis, or treatment.<br><br>
    Always consult a qualified healthcare provider for any health concerns. 
    Never ignore professional medical advice because of results from this system.
</div>
""", unsafe_allow_html=True)

st.divider()

# -------------------------------------------------
# Input form
# -------------------------------------------------
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=120,
        value=40,
        step=1,
        help="Patient's age in years"
    )
    sex = st.selectbox(
        "Sex",
        options=["Male", "Female"],
        help="Biological sex"
    )
    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=10.0,
        max_value=80.0,
        value=25.0,
        step=0.1,
        help="Weight (kg) / [Height (m)]²"
    )
    activity = st.selectbox(
        "Physical Activity",
        options=["No", "Yes"],
        help="Does the person engage in regular physical activity?"
    )

with col2:
    chol = st.selectbox(
        "High Cholesterol",
        options=["No", "Yes"],
        help="History of high cholesterol / hyperlipidemia"
    )
    diabetes = st.selectbox(
        "Diabetes Status",
        options=["No", "Yes"],
        help="Diagnosed with diabetes mellitus"
    )
    smoking = st.selectbox(
        "Smoking History",
        options=["No", "Yes"],
        help="Current or former smoker"
    )

st.markdown("")  # spacing

# -------------------------------------------------
# Prediction button
# -------------------------------------------------
predict_btn = st.button(
    "🔍 Predict Hypertension Risk",
    use_container_width=True,
    type="primary"
)

# -------------------------------------------------
# Prediction logic & results
# -------------------------------------------------
if predict_btn:
    # Prepare input in the exact feature order the model expects
    X_new = pd.DataFrame([{
        "RIDAGEYR": age,
        "RIAGENDR": 1 if sex == "Male" else 0,
        "BMXBMI": bmi,
        "BPQ080": 1 if chol == "Yes" else 0,
        "DIQ010": 1 if diabetes == "Yes" else 0,
        "SMQ020": 1 if smoking == "Yes" else 0,
        "Physical_Activity": 1 if activity == "Yes" else 0
    }])

    try:
        prediction = int(model.predict(X_new)[0])
        probabilities = model.predict_proba(X_new)[0]
        confidence = float(probabilities[prediction]) * 100

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-box hypertension">
                    <h2 style="color:#c62828; margin:0;">HYPERTENSION RISK</h2>
                    <p style="font-size:1.1rem; margin-top:0.5rem;">
                        The model indicates elevated risk of hypertension
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-box non-hypertension">
                    <h2 style="color:#2e7d32; margin:0;">NON-HYPERTENSION</h2>
                    <p style="font-size:1.1rem; margin-top:0.5rem;">
                        The model indicates lower risk of hypertension
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Confidence metric
        conf_col1, conf_col2, conf_col3 = st.columns([1, 2, 1])
        with conf_col2:
            st.metric(
                label="Model Confidence",
                value=f"{confidence:.1f}%"
            )

        # Probability breakdown
        with st.expander("View probability breakdown"):
            st.write(f"- Probability of **Non-Hypertension**: `{probabilities[0]*100:.1f}%`")
            st.write(f"- Probability of **Hypertension**: `{probabilities[1]*100:.1f}%`")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------------------------------
# PROMINENT DISCLAIMER (bottom)
# -------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ DISCLAIMER (Repeated for Safety)</strong><br><br>
    This system is intended solely for research and educational purposes. 
    It is <strong>not a diagnostic device</strong> and has not been approved 
    by any regulatory authority for clinical use.<br><br>
    Results should never replace consultation with a licensed physician or 
    qualified healthcare professional. If you have concerns about your blood 
    pressure or cardiovascular health, seek medical attention promptly.
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("""
<div class="footer">
    Random Forest Classifier &nbsp;|&nbsp; NHANES 2017–2018 &nbsp;|&nbsp; 7 Predictors<br>
    Age • Sex • BMI • Physical Activity • High Cholesterol • Smoking History • Diabetes Status
</div>
""", unsafe_allow_html=True)

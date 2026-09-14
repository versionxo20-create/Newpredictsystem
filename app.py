import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Hypertension Prediction System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}


/* =========================================================
   TOP HEADER
   ========================================================= */

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    margin-bottom: 1.2rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo-heart {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #fff0f2, #ffe1e6);
    font-size: 28px;
    box-shadow: 0 5px 16px rgba(0,0,0,0.08);
}

.brand-title {
    font-size: 1.25rem;
    font-weight: 700;
    line-height: 1.15;
}

.brand-subtitle {
    font-size: 0.78rem;
    opacity: 0.65;
    margin-top: 3px;
}


/* =========================================================
   BUTTONS
   ========================================================= */

div.stButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 600;
}


/* =========================================================
   NAVIGATION MENU
   ========================================================= */

.menu-box {
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 16px;
    padding: 12px;
    margin: 0 0 20px auto;
    max-width: 430px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.07);
}


/* =========================================================
   HOME HERO
   ========================================================= */

.hero {
    padding: 34px 28px;
    border-radius: 22px;
    background: linear-gradient(135deg, #fff5f6, #ffffff);
    border: 1px solid rgba(180,40,60,0.10);
    margin-bottom: 25px;
}

.hero h1 {
    margin: 0;
    font-size: 2.35rem;
    line-height: 1.15;
}

.hero p {
    margin-top: 12px;
    font-size: 1.02rem;
    opacity: 0.72;
    max-width: 760px;
}


/* =========================================================
   INFORMATION CARDS
   ========================================================= */

.info-card {
    border: 1px solid rgba(128,128,128,0.16);
    border-radius: 18px;
    padding: 22px;
    height: 100%;
    background: rgba(255,255,255,0.65);
    box-shadow: 0 6px 22px rgba(0,0,0,0.04);
}

.info-card h3 {
    margin-top: 0;
    font-size: 1.08rem;
}


/* =========================================================
   RESULT CARD
   ========================================================= */

.result-card {
    border-radius: 24px;
    padding: 30px 28px;
    border: 1px solid rgba(128,128,128,0.15);
    box-shadow: 0 10px 32px rgba(0,0,0,0.07);
    text-align: center;
    margin: 8px 0 24px;
}

.result-icon {
    font-size: 48px;
    margin-bottom: 5px;
}

.result-heading {
    font-size: 2rem;
    font-weight: 800;
    margin: 4px 0 8px;
}

.result-subtitle {
    opacity: 0.72;
    font-size: 1rem;
    margin-bottom: 20px;
}

.confidence-box {
    display: inline-block;
    padding: 12px 25px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.16);
    background: rgba(128,128,128,0.05);
}

.confidence-label {
    font-size: 0.76rem;
    opacity: 0.65;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.confidence-value {
    font-size: 1.55rem;
    font-weight: 800;
    margin-top: 3px;
}


/* =========================================================
   SECTION TITLE
   ========================================================= */

.section-title {
    font-size: 1.25rem;
    font-weight: 750;
    margin: 8px 0 14px;
}


/* =========================================================
   DISCLAIMER
   ========================================================= */

.disclaimer {
    padding: 15px 18px;
    border-radius: 14px;
    background: rgba(255,193,7,0.10);
    border: 1px solid rgba(255,193,7,0.25);
    font-size: 0.9rem;
    margin-top: 22px;
}


/* =========================================================
   FOOTER
   ========================================================= */

.custom-footer {
    text-align: center;
    opacity: 0.55;
    font-size: 0.78rem;
    margin-top: 42px;
    padding-top: 20px;
    border-top: 1px solid rgba(128,128,128,0.12);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("hypertension_random_forest.pkl")


model = load_model()


# ============================================================
# SESSION STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}

if "menu_open" not in st.session_state:
    st.session_state.menu_open = False


# ============================================================
# NAVIGATION FUNCTION
# ============================================================
def navigate(page):
    st.session_state.page = page
    st.session_state.menu_open = False
    st.rerun()


# ============================================================
# TOP HEADER
# ============================================================
st.markdown("""
<div class="topbar">

    <div class="brand">

        <div class="logo-heart">
            🫀
        </div>

        <div>
            <div class="brand-title">
                Hypertension Prediction System
            </div>

            <div class="brand-subtitle">
                Machine Learning Research System
            </div>
        </div>

    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# TOP-RIGHT NAVIGATION BUTTON
# ============================================================
left_space, right_space = st.columns([9, 1])

with right_space:

    if st.button(
        "☰",
        key="nav_toggle",
        use_container_width=True
    ):

        st.session_state.menu_open = not st.session_state.menu_open

        st.rerun()


# ============================================================
# NAVIGATION MENU
# ============================================================
if st.session_state.menu_open:

    st.markdown(
        '<div class="menu-box">',
        unsafe_allow_html=True
    )

    menu_col1, menu_col2 = st.columns(2)

    with menu_col1:

        if st.button(
            "⌂  Home",
            key="menu_home",
            use_container_width=True
        ):
            navigate("Home")

        if st.button(
            "▣  Model Information",
            key="menu_model",
            use_container_width=True
        ):
            navigate("Model")

    with menu_col2:

        if st.button(
            "⌁  Prediction",
            key="menu_prediction",
            use_container_width=True
        ):
            navigate("Prediction")

        if st.button(
            "ⓘ  About System",
            key="menu_about",
            use_container_width=True
        ):
            navigate("About")

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# HOME PAGE
# ============================================================
if st.session_state.page == "Home":

    st.markdown("""
    <div class="hero">

        <h1>
            Hypertension Prediction System
        </h1>

        <p>
            A machine learning research system designed to predict
            current hypertension status using selected health and
            lifestyle factors.
        </p>

    </div>
    """, unsafe_allow_html=True)


    card1, card2, card3 = st.columns(3)


    with card1:

        st.markdown("""
        <div class="info-card">

            <h3>
                🧠 Machine Learning
            </h3>

            <p>
                Uses a Random Forest classification model trained
                with NHANES 2017–2018 data.
            </p>

        </div>
        """, unsafe_allow_html=True)


    with card2:

        st.markdown("""
        <div class="info-card">

            <h3>
                📊 Seven Predictors
            </h3>

            <p>
                Age, sex, BMI, high cholesterol, diabetes,
                smoking history and physical activity.
            </p>

        </div>
        """, unsafe_allow_html=True)


    with card3:

        st.markdown("""
        <div class="info-card">

            <h3>
                ⚕️ Research Support
            </h3>

            <p>
                The result is intended for research and educational
                support and does not replace professional diagnosis.
            </p>

        </div>
        """, unsafe_allow_html=True)


    st.write("")


    if st.button(
        "Start New Prediction  →",
        key="home_predict",
        use_container_width=True
    ):

        navigate("Prediction")


# ============================================================
# PREDICTION PAGE
# ============================================================
elif st.session_state.page == "Prediction":

    st.markdown("## New Prediction")

    st.caption(
        "Enter the required information below to predict hypertension status."
    )


    with st.form("prediction_form"):

        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # LEFT SIDE
        # ----------------------------------------------------
        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=120,
                value=40,
                step=1
            )


            sex = st.selectbox(
                "Sex",
                ["Male", "Female"]
            )


            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=80.0,
                value=25.0,
                step=0.1
            )


            cholesterol = st.selectbox(
                "High Cholesterol",
                ["No", "Yes"]
            )


        # ----------------------------------------------------
        # RIGHT SIDE
        # ----------------------------------------------------
        with col2:

            diabetes = st.selectbox(
                "Diabetes",
                ["No", "Yes"]
            )


            smoking = st.selectbox(
                "Smoking History",
                ["No", "Yes"]
            )


            physical_activity = st.selectbox(
                "Physical Activity",
                ["No", "Yes"]
            )


        st.write("")


        button1, button2 = st.columns(2)


        with button1:

            predict_button = st.form_submit_button(
                "PREDICT STATUS",
                use_container_width=True
            )


        with button2:

            cancel_button = st.form_submit_button(
                "CANCEL",
                use_container_width=True
            )


    # --------------------------------------------------------
    # CANCEL
    # --------------------------------------------------------
    if cancel_button:

        navigate("Home")


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------
    if predict_button:

        input_data = pd.DataFrame([{

            "RIDAGEYR": age,

            "RIAGENDR":
                1 if sex == "Male" else 0,

            "BMXBMI": bmi,

            "BPQ080":
                1 if cholesterol == "Yes" else 0,

            "DIQ010":
                1 if diabetes == "Yes" else 0,

            "SMQ020":
                1 if smoking == "Yes" else 0,

            "Physical_Activity":
                1 if physical_activity == "Yes" else 0

        }])


        prediction = int(
            model.predict(input_data)[0]
        )


        if hasattr(model, "predict_proba"):

            confidence = float(
                max(
                    model.predict_proba(input_data)[0]
                ) * 100
            )

        else:

            confidence = None


        st.session_state.prediction = prediction

        st.session_state.confidence = confidence


        st.session_state.patient_data = {

            "Age": age,

            "Sex": sex,

            "BMI": bmi,

            "High Cholesterol": cholesterol,

            "Diabetes": diabetes,

            "Smoking History": smoking,

            "Physical Activity": physical_activity

        }


        navigate("Result")


# ============================================================
# RESULT PAGE
# ============================================================
elif st.session_state.page == "Result":

    prediction = st.session_state.prediction

    confidence = st.session_state.confidence

    data = st.session_state.patient_data


    st.markdown("## Prediction Result")

    st.caption(
        "Your submitted information has been evaluated by the machine learning model."
    )


    # --------------------------------------------------------
    # RESULT INFORMATION
    # --------------------------------------------------------
    if prediction == 1:

        icon = "⚠️"

        title = "HYPERTENSION"

        subtitle = (
            "The model predicts a positive hypertension status."
        )

        result_note = (
            "Consider discussing this result with a qualified "
            "healthcare professional."
        )

    else:

        icon = "✓"

        title = "NON-HYPERTENSION"

        subtitle = (
            "The model predicts a negative hypertension status."
        )

        result_note = (
            "This result does not rule out hypertension or replace "
            "a professional assessment."
        )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------
    confidence_html = ""


    if confidence is not None:

        confidence_html = f"""

        <div class="confidence-box">

            <div class="confidence-label">
                Model Confidence
            </div>

            <div class="confidence-value">
                {confidence:.2f}%
            </div>

        </div>

        """


    # --------------------------------------------------------
    # ATTRACTIVE RESULT CARD
    # --------------------------------------------------------
    st.markdown(
        f"""

        <div class="result-card">

            <div class="result-icon">
                {icon}
            </div>

            <div class="result-heading">
                {title}
            </div>

            <div class="result-subtitle">
                {subtitle}
            </div>

            {confidence_html}

        </div>

        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # NATIVE STREAMLIT MESSAGE
    # --------------------------------------------------------
    # This is intentionally native Streamlit rather than
    # rendering the result itself as raw HTML.
    if prediction == 1:

        st.warning(result_note)

    else:

        st.info(result_note)


    # --------------------------------------------------------
    # SUBMITTED INFORMATION
    # --------------------------------------------------------
    st.markdown(
        '<div class="section-title">Submitted Information</div>',
        unsafe_allow_html=True
    )


    result_df = pd.DataFrame({

        "Information":
            list(data.keys()),

        "Value":
            list(data.values())

    })


    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # MEDICAL DISCLAIMER
    # --------------------------------------------------------
    st.markdown("""
    <div class="disclaimer">

        <strong>
            Medical Disclaimer
        </strong>

        <br>

        This research doesn't replace professional medical diagnosis.
        Please visit a professional doctor for further examination.

    </div>
    """, unsafe_allow_html=True)


    st.write("")


    # --------------------------------------------------------
    # RESULT ACTION BUTTONS
    # --------------------------------------------------------
    result_button1, result_button2 = st.columns(2)


    with result_button1:

        if st.button(
            "← Back to Prediction",
            key="back_prediction",
            use_container_width=True
        ):

            navigate("Prediction")


    with result_button2:

        if st.button(
            "＋ New Prediction",
            key="new_prediction",
            use_container_width=True
        ):

            st.session_state.prediction = None

            st.session_state.confidence = None

            st.session_state.patient_data = {}

            navigate("Prediction")


# ============================================================
# MODEL INFORMATION PAGE
# ============================================================
elif st.session_state.page == "Model":

    st.markdown("## Model Information")

    st.caption(
        "Technical information about the predictive model used by the system."
    )


    metric1, metric2, metric3 = st.columns(3)


    with metric1:

        st.metric(
            "Usable Records",
            "5,224"
        )


    with metric2:

        st.metric(
            "Train / Test",
            "80% / 20%"
        )


    with metric3:

        st.metric(
            "Trees",
            "200"
        )


    st.write("")


    st.markdown("### Model Performance")


    metrics = pd.DataFrame({

        "Metric": [

            "Accuracy",

            "Precision",

            "Recall",

            "F1-Score",

            "ROC-AUC"

        ],

        "Result": [

            "71.29%",

            "60.91%",

            "54.03%",

            "57.26%",

            "76.57%"

        ]

    })


    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )


    st.markdown("### Feature Importance")


    importance = pd.DataFrame({

        "Predictor": [

            "BMI",

            "Age",

            "High Cholesterol",

            "Diabetes",

            "Smoking History",

            "Sex",

            "Physical Activity"

        ],

        "Importance": [

            "42.35%",

            "40.47%",

            "6.86%",

            "4.26%",

            "2.23%",

            "2.01%",

            "1.82%"

        ]

    })


    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABOUT SYSTEM PAGE
# ============================================================
elif st.session_state.page == "About":

    st.markdown("## About the System")


    st.mark

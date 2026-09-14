import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hypertension Prediction System",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Remove Streamlit default elements */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Main page spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 900px;
    }

    /* Main title */
    .app-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 0px;
        line-height: 1.2;
    }

    .app-subtitle {
        font-size: 14px;
        opacity: 0.65;
        margin-top: 3px;
    }

    /* Navigation menu */
    .nav-box {
        padding: 12px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 5px;
        margin-bottom: 15px;
    }

    /* Home hero */
    .hero-title {
        font-size: 38px;
        font-weight: 750;
        line-height: 1.1;
        margin-bottom: 12px;
    }

    .hero-text {
        font-size: 17px;
        line-height: 1.6;
        opacity: 0.8;
    }

    /* Cards */
    .info-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.20);
        margin-bottom: 15px;
        min-height: 130px;
    }

    .info-card h3 {
        margin-top: 0px;
        margin-bottom: 8px;
    }

    .info-card p {
        opacity: 0.75;
        line-height: 1.5;
    }

    /* Section heading */
    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        opacity: 0.55;
        font-size: 13px;
        margin-top: 35px;
        padding-top: 15px;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("hypertension_random_forest.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the prediction model.")
    st.info(
        "Please make sure 'hypertension_random_forest.pkl' "
        "is in the same folder as app.py."
    )
    st.stop()


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
    st.session_state.patient_data = None

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

header_left, header_right = st.columns([8.5, 1.5])

with header_left:
    st.markdown(
        '<div class="app-title">🫀 Hypertension Prediction System</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="app-subtitle">Machine Learning Research System</div>',
        unsafe_allow_html=True
    )

with header_right:
    st.write("")
    if st.button(
        "☰",
        key="nav_toggle",
        help="Open navigation",
        use_container_width=True
    ):
        st.session_state.menu_open = not st.session_state.menu_open
        st.rerun()


# ============================================================
# NAVIGATION MENU
# ============================================================

if st.session_state.menu_open:

    st.markdown('<div class="nav-box">', unsafe_allow_html=True)

    st.markdown("### Navigation")

    nav1, nav2 = st.columns(2)

    with nav1:
        if st.button("🏠 Home", use_container_width=True):
            navigate("Home")

        if st.button("🧠 Prediction", use_container_width=True):
            navigate("Prediction")

    with nav2:
        if st.button("📊 Model Information", use_container_width=True):
            navigate("Model Information")

        if st.button("ℹ️ About System", use_container_width=True):
            navigate("About System")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "Home":

    st.write("")

    st.markdown(
        '<div class="hero-title">Hypertension Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-text">
        A machine learning research system designed to classify
        hypertension status using selected health and lifestyle
        characteristics.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "🧠 Start Prediction",
        use_container_width=True,
        type="primary"
    ):
        navigate("Prediction")

    st.write("")

    card1, card2 = st.columns(2)

    with card1:
        st.markdown(
            """
            <div class="info-card">
                <h3>🔬 Machine Learning</h3>
                <p>
                The system uses a Random Forest classification model
                trained using NHANES 2017–2018 data.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with card2:
        st.markdown(
            """
            <div class="info-card">
                <h3>📋 Seven Predictors</h3>
                <p>
                Age, sex, BMI, high cholesterol, diabetes,
                smoking history and physical activity.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.info(
        "This system is intended for research and educational purposes "
        "and does not replace professional medical diagnosis."
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif st.session_state.page == "Prediction":

    st.markdown(
        '<div class="section-title">Patient Information</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the required information below to classify the "
        "hypertension status."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

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

        submitted = st.form_submit_button(
            "🔍 Predict Hypertension Status",
            use_container_width=True,
            type="primary"
        )

    if submitted:

        # --------------------------------------------------------
        # Convert user inputs to model format
        # --------------------------------------------------------

        sex_value = 1 if sex == "Male" else 0
        cholesterol_value = 1 if cholesterol == "Yes" else 0
        diabetes_value = 1 if diabetes == "Yes" else 0
        smoking_value = 1 if smoking == "Yes" else 0
        physical_activity_value = (
            1 if physical_activity == "Yes" else 0
        )

        # --------------------------------------------------------
        # Create dataframe in the same feature order used for
        # model training
        # --------------------------------------------------------

        input_data = pd.DataFrame({
            "Age": [age],
            "Sex": [sex_value],
            "BMI": [bmi],
            "High_Cholesterol": [cholesterol_value],
            "Diabetes": [diabetes_value],
            "Smoking_History": [smoking_value],
            "Physical_Activity": [physical_activity_value]
        })

        # --------------------------------------------------------
        # Prediction
        # --------------------------------------------------------

        prediction = model.predict(input_data)[0]

        # Get prediction probability where supported
        try:
            probabilities = model.predict_proba(input_data)[0]
            confidence = max(probabilities) * 100
        except Exception:
            confidence = None

        # --------------------------------------------------------
        # Save result
        # --------------------------------------------------------

        st.session_state.prediction = int(prediction)
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
    patient_data = st.session_state.patient_data

    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # ========================================================
    # RESULT CARD
    # ========================================================

    result_left, result_right = st.columns([2.5, 1.5])

    with result_left:

        if prediction == 1:

            st.error("⚠️ HYPERTENSION")

            st.markdown(
                "### The model predicts a positive hypertension status."
            )

            st.warning(
                "Consider discussing this result with a qualified "
                "healthcare professional."
            )

        else:

            st.success("✓ NON-HYPERTENSION")

            st.markdown(
                "### The model predicts a negative hypertension status."
            )

            st.info(
                "This result does not rule out hypertension or replace "
                "a professional medical assessment."
            )

    with result_right:

        if confidence is not None:

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

    st.divider()

    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.markdown("### Submitted Information")

    if patient_data is not None:

        result_table = pd.DataFrame(
            list(patient_data.items()),
            columns=["Variable", "Value"]
        )

        st.dataframe(
            result_table,
            use_container_width=True,
            hide_index=True
        )

    st.write("")

    # ========================================================
    # MEDICAL DISCLAIMER
    # ========================================================

    st.warning(
        "Medical Disclaimer\n\n"
        "This research doesn't replace professional medical diagnosis. "
        "Please visit a professional doctor for further examination."
    )

    st.write("")

    # ========================================================
    # RESULT ACTIONS
    # ========================================================

    action1, action2 = st.columns(2)

    with action1:

        if st.button(
            "🔄 New Prediction",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.prediction = None
            st.session_state.confidence = None
            st.session_state.patient_data = None
            navigate("Prediction")

    with action2:

        if st.button(
            "🏠 Back to Home",
            use_container_width=True
        ):
            navigate("Home")


# ============================================================
# MODEL INFORMATION PAGE
# ============================================================

elif st.session_state.page == "Model Information":

    st.markdown(
        '<div class="section-title">Model Information</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The hypertension prediction system uses a Random Forest "
        "classification algorithm trained using NHANES 2017–2018 data."
    )

    st.write("")

    st.markdown("### Dataset")

    dataset_info = pd.DataFrame({
        "Item": [
            "Dataset",
            "Data Source",
            "Usable Records",
            "Input Variables",
            "Algorithm",
            "Train/Test Split"
        ],
        "Details": [
            "NHANES 2017–2018",
            "National Health and Nutrition Examination Survey",
            "5,224",
            "7",
            "Random Forest",
            "80% / 20%"
        ]
    })

    st.dataframe(
        dataset_info,
        use_container_width=True,
        hide_index=True
    )

    st.write("")

    st.markdown("### Model Performance")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric("Accuracy", "71.29%")
        st.metric("Recall", "54.03%")
        st.metric("ROC-AUC", "76.57%")

    with metric2:
        st.metric("Precision", "60.91%")
        st.metric("F1 Score", "57.26%")

    st.write("")

    st.markdown("### Feature Importance")

    feature_importance = pd.DataFrame({
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
        feature_importance,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The model is a research/educational classification system. "
        "Its output should not be interpreted as a medical diagnosis."
    )


# ============================================================
# ABOUT SYSTEM PAGE
# ============================================================

elif st.session_state.page == "About System":

    st.markdown(
        '<div class="section-title">About the System</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The Hypertension Prediction System is a research-based "
        "machine learning application developed to classify "
        "hypertension status from selected health and lifestyle "
        "characteristics."
    )

    st.write("")

    st.markdown("### Predictors Used")

    predictors = [
        "Age",
        "Sex",
        "BMI",
        "High Cholesterol",
        "Diabetes",
        "Smoking History",
        "Physical Activity"
    ]

    for predictor in predictors:
        st.write(f"• {predictor}")

    st.write("")

    st.markdown("### Algorithm")

    st.write(
        "Random Forest was selected as the primary machine learning "
        "algorithm because it combines multiple decision trees to "
        "produce a classification result and can capture "
        "relationships among several input variables."
    )

    st.write("")

    st.markdown("### Purpose")

    st.write(
        "The system demonstrates how machine learning can be applied "
        "to health-related data for hypertension status classification."
    )

    st.warning(
        "This research doesn't replace professional medical diagnosis. "
        "Please visit a professional doctor for further examination."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="custom-footer">
        © 2026 Hypertension Prediction System<br>
        Research and Educational Purpose
    </div>
    """,
    unsafe_allow_html=True
)

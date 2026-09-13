import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hypertension Prediction System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM DESIGN
# =========================================================

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
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 1100px;
}


/* LOGO */

.logo-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 15px;
    margin-top: 10px;
    margin-bottom: 8px;
}

.logo-heart {
    font-size: 55px;
    line-height: 1;
}

.logo-title {
    font-size: 30px;
    font-weight: 800;
}


/* SUBTITLE */

.page-subtitle {
    text-align: center;
    font-size: 17px;
    opacity: 0.75;
    margin-bottom: 30px;
}


/* CARDS */

.info-card {
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.25);
    min-height: 165px;
    margin-bottom: 20px;
}

.info-card h3 {
    margin-top: 0;
}


/* RESULT CARD */

.result-card {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid rgba(128,128,128,0.25);
    margin-top: 20px;
    margin-bottom: 25px;
}

.result-title {
    font-size: 34px;
    font-weight: 800;
}


/* SECTION */

.section-title {
    font-size: 25px;
    font-weight: 750;
    margin-top: 15px;
    margin-bottom: 15px;
}


/* BUTTONS */

div.stButton > button {
    border-radius: 11px;
    min-height: 46px;
    font-weight: 650;
}


/* DISCLAIMER */

.disclaimer {
    padding: 18px;
    border-radius: 13px;
    border: 1px solid rgba(255, 170, 0, 0.45);
    margin-top: 25px;
    font-size: 14px;
}


/* COPYRIGHT */

.copyright {
    text-align: center;
    margin-top: 55px;
    padding-top: 18px;
    border-top: 1px solid rgba(128,128,128,0.25);
    font-size: 13px;
    opacity: 0.65;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = Path(__file__).parent / "hypertension_random_forest.pkl"


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# SESSION STATE
# =========================================================

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


# =========================================================
# NAVIGATION FUNCTION
# =========================================================

def navigate(page):
    st.session_state.page = page
    st.session_state.menu_open = False
    st.rerun()


# =========================================================
# NAVIGATION BUTTON
# =========================================================

if st.button(
    "☰",
    use_container_width=False
):
    st.session_state.menu_open = not st.session_state.menu_open
    st.rerun()


# =========================================================
# NAVIGATION MENU
# =========================================================

if st.session_state.menu_open:

    st.markdown("### Menu")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "⌂  Home",
            use_container_width=True
        ):
            navigate("Home")

        if st.button(
            "▣  Model Information",
            use_container_width=True
        ):
            navigate("Model")

    with col2:

        if st.button(
            "⌁  Prediction",
            use_container_width=True
        ):
            navigate("Prediction")

        if st.button(
            "ⓘ  About System",
            use_container_width=True
        ):
            navigate("About")

    st.divider()


# =========================================================
# HEADER / LOGO
# =========================================================

st.markdown(
    """
    <div class="logo-container">
        <div class="logo-heart">🫀</div>
        <div class="logo-title">
            Hypertension Prediction System
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HOME PAGE
# =========================================================

if st.session_state.page == "Home":

    st.markdown(
        """
        <div class="page-subtitle">
        Machine learning-based prediction of hypertension status
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="info-card">
                <h3>🩺 Hypertension Prediction</h3>
                <p>
                This system predicts hypertension status using
                selected health and lifestyle information.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">
                <h3>🤖 Machine Learning</h3>
                <p>
                A Random Forest classification model trained
                with NHANES 2017–2018 data is used to generate
                the prediction.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset",
            "NHANES 2017–2018"
        )

    with col2:
        st.metric(
            "Predictors",
            "7"
        )

    with col3:
        st.metric(
            "Algorithm",
            "Random Forest"
        )

    st.divider()

    st.subheader("Start a Prediction")

    st.write(
        "Enter health and lifestyle information to obtain "
        "a predicted hypertension status."
    )

    if st.button(
        "START PREDICTION  →",
        use_container_width=True
    ):
        navigate("Prediction")

    st.markdown(
        """
        <div class="disclaimer">
        <b>Disclaimer</b><br><br>
        This research doesn't replace professional medical diagnosis.
        Please visit a professional doctor for further examination.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PREDICTION PAGE
# =========================================================

elif st.session_state.page == "Prediction":

    st.title("Patient Information")

    st.write(
        "Enter the required information below."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age (years)",
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
            "Body Mass Index (BMI)",
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

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        predict_button = st.button(
            "PREDICT STATUS",
            use_container_width=True
        )

    with col2:

        cancel_button = st.button(
            "CANCEL",
            use_container_width=True
        )

    if cancel_button:
        navigate("Home")


    # =====================================================
    # MAKE PREDICTION
    # =====================================================

    if predict_button:

        input_data = pd.DataFrame([
            {
                "RIDAGEYR": age,
                "RIAGENDR": 1 if sex == "Male" else 0,
                "BMXBMI": bmi,
                "BPQ080": 1 if cholesterol == "Yes" else 0,
                "DIQ010": 1 if diabetes == "Yes" else 0,
                "SMQ020": 1 if smoking == "Yes" else 0,
                "Physical_Activity":
                    1 if physical_activity == "Yes" else 0
            }
        ])

        prediction = int(
            model.predict(input_data)[0]
        )

        probabilities = model.predict_proba(
            input_data
        )[0]

        confidence = probabilities[prediction] * 100

        st.session_state.prediction = prediction
        st.session_state.confidence = confidence

        st.session_state.patient_data = {
            "Age": age,
            "Sex": sex,
            "BMI": f"{bmi:.1f}",
            "High Cholesterol": cholesterol,
            "Diabetes": diabetes,
            "Smoking History": smoking,
            "Physical Activity": physical_activity
        }

        navigate("Result")


# =========================================================
# RESULT PAGE
# =========================================================

elif st.session_state.page == "Result":

    st.title("Prediction Result")

    st.write(
        "The Random Forest model has completed the prediction."
    )

    st.divider()

    prediction = st.session_state.prediction
    confidence = st.session_state.confidence


    # =====================================================
    # RESULT
    # =====================================================

    if prediction == 1:

        st.error(
            "HYPERTENSION"
        )

        st.write(
            "The model predicts a positive hypertension status."
        )

    else:

        st.success(
            "NON-HYPERTENSION"
        )

        st.write(
            "The model predicts a negative hypertension status."
        )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    st.subheader("Model Confidence")

    st.progress(
        min(int(confidence), 100)
    )

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )


    # =====================================================
    # PATIENT INFORMATION
    # =====================================================

    st.divider()

    st.subheader("Submitted Information")

    patient = st.session_state.patient_data

    result_table = pd.DataFrame({
        "Variable": list(patient.keys()),
        "Value": list(patient.values())
    })

    st.dataframe(
        result_table,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # RESULT BUTTONS
    # =====================================================

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "← BACK TO PREDICTION",
            use_container_width=True
        ):
            navigate("Prediction")

    with col2:

        if st.button(
            "＋ NEW PREDICTION",
            use_container_width=True
        ):

            st.session_state.prediction = None
            st.session_state.confidence = None
            st.session_state.patient_data = None

            navigate("Prediction")


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.markdown(
        """
        <div class="disclaimer">
        <b>Medical Disclaimer</b><br><br>
        This research doesn't replace professional medical diagnosis.
        Please visit a professional doctor for further examination.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MODEL INFORMATION PAGE
# =========================================================

elif st.session_state.page == "Model":

    st.title("Model Information")

    st.write(
        "Technical information about the machine learning model "
        "used in the system."
    )

    st.divider()

    st.subheader("Random Forest Classifier")

    st.write(
        "The Random Forest algorithm was selected as the primary "
        "predictive algorithm for the system."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Usable Records", "5,224")

    with col2:
        st.metric("Train / Test", "80 / 20")

    with col3:
        st.metric("Trees", "200")

    st.divider()

    st.subheader("Predictor Variables")

    predictors = pd.DataFrame({
        "No.": [1, 2, 3, 4, 5, 6, 7],

        "Predictor": [
            "Age",
            "Sex",
            "Body Mass Index (BMI)",
            "High Cholesterol",
            "Diabetes",
            "Smoking History",
            "Physical Activity"
        ]
    })

    st.dataframe(
        predictors,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Model Performance")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Accuracy", "71.29%")

    with col2:
        st.metric("Precision", "60.91%")

    with col3:
        st.metric("Recall", "54.03%")

    with col4:
        st.metric("F1-Score", "57.26%")

    with col5:
        st.metric("ROC-AUC", "76.57%")

    st.divider()

    st.subheader("Feature Importance")

    features = pd.DataFrame({
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
        features,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# ABOUT PAGE
# =========================================================

elif st.session_state.page == "About":

    st.title("About the System")

    st.write(
        "The Hypertension Prediction System is a research-based "
        "machine learning application developed to predict "
        "hypertension status using selected health and lifestyle "
        "variables."
    )

    st.divider()

    st.subheader("Project Objective")

    st.write(
        "The objective of the system is to design and implement "
        "a machine learning-based system capable of predicting "
        "hypertension status from selected health and lifestyle "
        "variables."
    )

    st.subheader("Dataset")

    st.write(
        "The system was developed using the NHANES 2017–2018 dataset."
    )

    st.subheader("Technologies Used")

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "Scikit-learn",
            "Joblib",
            "Streamlit"
        ],

        "Purpose": [
            "Programming",
            "Data processing",
            "Machine learning",
            "Model storage",
            "Web application"
        ]
    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown(
        """
        <div class="disclaimer">
        <b>Medical Disclaimer</b><br><br>
        This research doesn't replace professional medical diagnosis.
        Please visit a professional doctor for further examination.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# COPYRIGHT
# =========================================================

st.markdown(
    """
    <div class="copyright">
        © 2026 Hypertension Prediction System. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)

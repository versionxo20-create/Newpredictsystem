import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hypertension Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# HIDE STREAMLIT DEFAULT FOOTER AND MENU
# =========================================================

st.markdown(
    """
    <style>
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
    }

    .info-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = Path(__file__).parent / "hypertension_random_forest.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("❤️ Hypertension System")

st.sidebar.write("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Prediction",
        "Model Information",
        "About System"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Random Forest Model\n\n"
    "NHANES 2017–2018\n\n"
    "7 Predictor Variables"
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "Home":

    st.markdown(
        '<div class="main-title">Hypertension Prediction System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'A machine learning-based system for predicting hypertension status'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🤖 Machine Learning")
        st.write(
            "The system uses a Random Forest classification "
            "algorithm to predict hypertension status."
        )

    with col2:
        st.subheader("📊 NHANES Dataset")
        st.write(
            "The model was developed using data from the "
            "NHANES 2017–2018 dataset."
        )

    with col3:
        st.subheader("🩺 Health Information")
        st.write(
            "The system uses selected health and lifestyle "
            "variables to generate a prediction."
        )

    st.divider()

    st.subheader("How to Use the System")

    st.write("1. Go to the **Prediction** page.")
    st.write("2. Enter the required health information.")
    st.write("3. Click **Predict Hypertension Status**.")
    st.write("4. View the prediction and model confidence.")

    st.divider()

    st.warning(
        "This research system is intended for research and "
        "educational purposes and should not be used as a "
        "substitute for professional medical diagnosis."
    )


# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "Prediction":

    st.title("🔮 Hypertension Prediction")

    st.write(
        "Enter the required information below to predict "
        "the individual's hypertension status."
    )

    st.divider()

    # -----------------------------------------------------
    # PATIENT INFORMATION
    # -----------------------------------------------------

    st.subheader("Patient Information")

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

    # -----------------------------------------------------
    # BUTTONS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        predict_button = st.button(
            "🔮 Predict Hypertension Status",
            use_container_width=True
        )

    with col2:

        new_prediction_button = st.button(
            "🔄 New Prediction",
            use_container_width=True
        )

    # -----------------------------------------------------
    # NEW PREDICTION / RESET
    # -----------------------------------------------------

    if new_prediction_button:

        st.session_state.prediction_made = False

        st.rerun()

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if predict_button:

        input_data = pd.DataFrame(
            [
                {
                    "RIDAGEYR": age,
                    "RIAGENDR": 1 if sex == "Male" else 0,
                    "BMXBMI": bmi,
                    "BPQ080": 1 if cholesterol == "Yes" else 0,
                    "DIQ010": 1 if diabetes == "Yes" else 0,
                    "SMQ020": 1 if smoking == "Yes" else 0,
                    "Physical_Activity": (
                        1 if physical_activity == "Yes" else 0
                    )
                }
            ]
        )

        prediction = int(
            model.predict(input_data)[0]
        )

        probability = model.predict_proba(
            input_data
        )[0]

        confidence = probability[prediction] * 100

        st.session_state.prediction_made = True

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                "⚠️ HYPERTENSION"
            )

            st.write(
                "The model predicts a positive hypertension status."
            )

        else:

            st.success(
                "✓ NON-HYPERTENSION"
            )

            st.write(
                "The model predicts a negative hypertension status."
            )

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.divider()

        st.subheader("Entered Information")

        result_data = pd.DataFrame(
            {
                "Variable": [
                    "Age",
                    "Sex",
                    "BMI",
                    "High Cholesterol",
                    "Diabetes",
                    "Smoking History",
                    "Physical Activity"
                ],
                "Value": [
                    age,
                    sex,
                    f"{bmi:.1f}",
                    cholesterol,
                    diabetes,
                    smoking,
                    physical_activity
                ]
            }
        )

        st.table(result_data)

        st.divider()

        st.warning(
            "This research doesn't replace professional medical "
            "diagnosis. Please visit a professional doctor for "
            "further examination."
        )


# =========================================================
# MODEL INFORMATION PAGE
# =========================================================

elif page == "Model Information":

    st.title("📊 Model Information")

    st.write(
        "The prediction system uses a Random Forest classification "
        "model trained using the NHANES 2017–2018 dataset."
    )

    st.divider()

    st.subheader("Algorithm")

    st.write("**Random Forest Classifier**")

    st.write(
        "Random Forest is an ensemble machine learning algorithm "
        "that combines multiple decision trees to produce a "
        "classification result."
    )

    st.subheader("Dataset")

    st.write(
        "**National Health and Nutrition Examination Survey "
        "(NHANES) 2017–2018**"
    )

    st.write(
        "After data preprocessing and removal of records with "
        "missing required values, 5,224 usable records were "
        "available for model development."
    )

    st.divider()

    st.subheader("Predictor Variables")

    predictors = pd.DataFrame(
        {
            "No.": [1, 2, 3, 4, 5, 6, 7],
            "Predictor Variable": [
                "Age",
                "Sex",
                "Body Mass Index (BMI)",
                "High Cholesterol",
                "Diabetes",
                "Smoking History",
                "Physical Activity"
            ]
        }
    )

    st.table(predictors)

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

    feature_data = pd.DataFrame(
        {
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
        }
    )

    st.table(feature_data)


# =========================================================
# ABOUT SYSTEM PAGE
# =========================================================

elif page == "About System":

    st.title("ℹ️ About the System")

    st.write(
        "The Hypertension Prediction System is a research-based "
        "machine learning application developed to predict "
        "hypertension status using selected health and lifestyle "
        "information."
    )

    st.divider()

    st.subheader("Purpose")

    st.write(
        "The purpose of the system is to demonstrate how machine "
        "learning can be applied to health-related data to predict "
        "hypertension status."
    )

    st.subheader("Technology Used")

    technologies = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Pandas",
                "Scikit-learn",
                "Joblib",
                "Streamlit",
                "NHANES 2017–2018"
            ],
            "Purpose": [
                "Programming language",
                "Data processing",
                "Machine learning",
                "Model storage and loading",
                "Web application interface",
                "Dataset"
            ]
        }
    )

    st.table(technologies)

    st.divider()

    st.subheader("Important Disclaimer")

    st.warning(
        "This research doesn't replace professional medical "
        "diagnosis. Please visit a professional doctor for "
        "further examination."
    )

    st.divider()

    st.caption(
        "Hypertension Prediction System | "
        "Random Forest | NHANES 2017–2018"
    )

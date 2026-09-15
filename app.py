import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


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
# HIDE STREAMLIT DEFAULT ELEMENTS
# ============================================================

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 45px;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = Path(__file__).parent / "hypertension_random_forest.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "input_data" not in st.session_state:
    st.session_state.input_data = None


# ============================================================
# NAVIGATION FUNCTION
# ============================================================

def navigate(page_name):
    st.session_state.page = page_name


# ============================================================
# TOP HEADER
# ============================================================

header_left, header_right = st.columns([8, 1])

with header_left:
    st.title("🫀 Hypertension Prediction System")
    st.caption("Machine Learning Research System")

with header_right:
    st.write("")
    st.write("")
    if st.button("☰", key="menu_button", use_container_width=True):
        st.session_state.show_menu = not st.session_state.get(
            "show_menu", False
        )


# ============================================================
# NAVIGATION MENU
# ============================================================

if st.session_state.get("show_menu", False):

    st.divider()

    st.subheader("Navigation")

    menu_col1, menu_col2 = st.columns(2)

    with menu_col1:
        if st.button("🏠 Home", use_container_width=True):
            navigate("Home")
            st.session_state.show_menu = False
            st.rerun()

        if st.button("🔮 Prediction", use_container_width=True):
            navigate("Prediction")
            st.session_state.show_menu = False
            st.rerun()

    with menu_col2:
        if st.button("📊 Model Information", use_container_width=True):
            navigate("Model Information")
            st.session_state.show_menu = False
            st.rerun()

        if st.button("ℹ️ About System", use_container_width=True):
            navigate("About")
            st.session_state.show_menu = False
            st.rerun()

    st.divider()


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "Home":

    st.write("")

    st.info(
        "A machine learning research system designed to classify "
        "current hypertension status using selected health and "
        "lifestyle factors."
    )

    st.write("")

    st.subheader("System Overview")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("🧠 Machine Learning")
            st.write(
                "Uses a Random Forest classification model trained "
                "with NHANES 2017–2018 data."
            )

    with col2:
        with st.container(border=True):
            st.subheader("📋 Seven Predictors")
            st.write(
                "Age, sex, BMI, high cholesterol, diabetes, "
                "smoking history and physical activity."
            )

    st.write("")

    col3, col4 = st.columns(2)

    with col3:
        with st.container(border=True):
            st.subheader("🎯 Prediction")
            st.write(
                "Classifies the entered information as "
                "hypertension or non-hypertension."
            )

    with col4:
        with st.container(border=True):
            st.subheader("📊 Model Performance")
            st.write(
                "The model achieved an ROC-AUC of 76.57% "
                "on the test data."
            )

    st.write("")

    if st.button(
        "🔮 Start Hypertension Prediction",
        use_container_width=True
    ):
        navigate("Prediction")
        st.rerun()

    st.write("")

    st.warning(
        "This research doesn't replace professional medical diagnosis. "
        "Please visit a professional doctor for further examination."
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif st.session_state.page == "Prediction":

    st.header("🔮 Hypertension Prediction")

    st.write(
        "Enter the required information below to classify "
        "the current hypertension status."
    )

    if not model_loaded:
        st.error("The prediction model could not be loaded.")
        st.code(model_error)
        st.stop()

    st.divider()

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=120,
            value=40,
            step=1
        )

    with col2:
        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

    # --------------------------------------------------------
    # HEALTH INFORMATION
    # --------------------------------------------------------

    st.subheader("Health Information")

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=80.0,
        value=25.0,
        step=0.1
    )

    col3, col4 = st.columns(2)

    with col3:
        cholesterol = st.selectbox(
            "High Cholesterol",
            ["No", "Yes"]
        )

    with col4:
        diabetes = st.selectbox(
            "Diabetes",
            ["No", "Yes"]
        )

    # --------------------------------------------------------
    # LIFESTYLE INFORMATION
    # --------------------------------------------------------

    st.subheader("Lifestyle Information")

    col5, col6 = st.columns(2)

    with col5:
        smoking = st.selectbox(
            "Smoking History",
            ["No", "Yes"]
        )

    with col6:
        physical_activity = st.selectbox(
            "Physical Activity",
            ["No", "Yes"]
        )

    st.write("")

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Hypertension Status",
        type="primary",
        use_container_width=True
    ):

        try:

            # Keep the feature names consistent with the trained model.
            # The categorical variables use the original NHANES coding.

            input_data = pd.DataFrame({
                "RIDAGEYR": [age],
                "RIAGENDR": [
                    1 if sex == "Male" else 2
                ],
                "BMXBMI": [bmi],
                "BPQ080": [
                    1 if cholesterol == "Yes" else 2
                ],
                "DIQ010": [
                    1 if diabetes == "Yes" else 2
                ],
                "SMQ020": [
                    1 if smoking == "Yes" else 2
                ],
                "Physical_Activity": [
                    1 if physical_activity == "Yes" else 0
                ]
            })

            prediction = int(model.predict(input_data)[0])

            confidence = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                confidence = float(max(probabilities)) * 100

            st.session_state.prediction = prediction
            st.session_state.confidence = confidence
            st.session_state.input_data = input_data

            navigate("Result")

            st.rerun()

        except Exception as e:

            st.error("Prediction could not be completed.")

            st.write(
                "Please check that the model file matches "
                "the application."
            )

            st.code(str(e))


# ============================================================
# RESULT PAGE
# ============================================================

elif st.session_state.page == "Result":

    st.header("📊 Prediction Result")

    prediction = st.session_state.prediction
    confidence = st.session_state.confidence

    if prediction is None:

        st.info(
            "No prediction has been made yet."
        )

        if st.button(
            "🔮 Make a Prediction",
            use_container_width=True
        ):
            navigate("Prediction")
            st.rerun()

    else:

        st.write("")

        # ----------------------------------------------------
        # RESULT DISPLAY
        # ----------------------------------------------------

        if prediction == 1:

            st.error("### HYPERTENSION")

            st.write(
                "The model predicts a positive hypertension status "
                "based on the information entered."
            )

        else:

            st.success("### NON-HYPERTENSION")

            st.write(
                "The model predicts a negative hypertension status "
                "based on the information entered."
            )

        st.write("")

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if confidence is not None:

            st.subheader("Prediction Confidence")

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(
                min(max(confidence / 100, 0.0), 1.0)
            )

        st.write("")

        # ----------------------------------------------------
        # ENTERED INFORMATION
        # ----------------------------------------------------

        st.subheader("Entered Information")

        if st.session_state.input_data is not None:

            data = st.session_state.input_data.iloc[0]

            result_table = pd.DataFrame({
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
                    int(data["RIDAGEYR"]),
                    "Male" if data["RIAGENDR"] == 1 else "Female",
                    f"{data['BMXBMI']:.1f}",
                    "Yes" if data["BPQ080"] == 1 else "No",
                    "Yes" if data["DIQ010"] == 1 else "No",
                    "Yes" if data["SMQ020"] == 1 else "No",
                    "Yes" if data["Physical_Activity"] == 1 else "No"
                ]
            })

            st.dataframe(
                result_table,
                use_container_width=True,
                hide_index=True
            )

        st.write("")

        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------

        st.warning(
            "This research doesn't replace professional medical diagnosis. "
            "Please visit a professional doctor for further examination."
        )

        st.write("")

        # ----------------------------------------------------
        # NEW PREDICTION
        # ----------------------------------------------------

        if st.button(
            "🔄 New Prediction",
            type="primary",
            use_container_width=True
        ):

            st.session_state.prediction = None
            st.session_state.confidence = None
            st.session_state.input_data = None

            navigate("Prediction")

            st.rerun()


# ============================================================
# MODEL INFORMATION PAGE
# ============================================================

elif st.session_state.page == "Model Information":

    st.header("📊 Model Information")

    st.write(
        "The system uses a Random Forest classification algorithm "
        "to classify current hypertension status."
    )

    st.divider()

    st.subheader("Dataset")

    st.write(
        "NHANES 2017–2018 "
        "(National Health and Nutrition Examination Survey)"
    )

    st.write("Final usable records: **5,224**")

    st.write("")

    st.subheader("Algorithm")

    with st.container(border=True):
        st.subheader("🌳 Random Forest")
        st.write(
            "Random Forest is an ensemble machine learning algorithm "
            "that combines multiple decision trees to make a "
            "classification decision."
        )

    st.write("")

    st.subheader("Seven Predictors")

    predictors = pd.DataFrame({
        "No.": [1, 2, 3, 4, 5, 6, 7],
        "Predictor": [
            "Age",
            "Sex",
            "BMI",
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

    st.write("")

    st.subheader("Model Performance")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            "Accuracy",
            "71.29%"
        )

        st.metric(
            "Recall",
            "54.03%"
        )

        st.metric(
            "ROC-AUC",
            "76.57%"
        )

    with metric2:
        st.metric(
            "Precision",
            "60.91%"
        )

        st.metric(
            "F1-Score",
            "57.26%"
        )

    st.write("")

    st.info(
        "The model performance represents the evaluation obtained "
        "from the test dataset and should not be interpreted as "
        "clinical diagnostic accuracy."
    )


# ============================================================
# ABOUT PAGE
# ============================================================

elif st.session_state.page == "About":

    st.header("ℹ️ About the System")

    st.write(
        "The Hypertension Prediction System is a research-based "
        "machine learning application developed to classify "
        "current hypertension status."
    )

    st.write(
        "The system uses selected demographic, health and lifestyle "
        "variables from the NHANES 2017–2018 dataset."
    )

    st.divider()

    st.subheader("Purpose")

    st.write(
        "The purpose of the system is to demonstrate how machine "
        "learning can be applied to health-related data to classify "
        "hypertension status."
    )

    st.subheader("Data Source")

    st.write(
        "National Health and Nutrition Examination Survey "
        "(NHANES), 2017–2018."
    )

    st.subheader("Machine Learning Algorithm")

    st.write(
        "Random Forest Classifier."
    )

    st.subheader("Important Notice")

    st.warning(
        "This research doesn't replace professional medical diagnosis. "
        "Please visit a professional doctor for further examination."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "© 2026 Hypertension Prediction System | "
    "Machine Learning Research Project"
)

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Hypertension Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
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
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}


/* Main title */

.hero-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 8px;
}

.hero-subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}


/* Navigation */

.nav-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}


/* Cards */

.card {
    padding: 24px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 18px;
    min-height: 150px;
}

.card-title {
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 8px;
}

.card-text {
    font-size: 15px;
}


/* Result */

.result-box {
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(128,128,128,0.25);
    margin-top: 20px;
    margin-bottom: 20px;
}

.result-title {
    font-size: 32px;
    font-weight: 800;
}


/* Section */

.section-title {
    font-size: 26px;
    font-weight: 750;
    margin-top: 15px;
}


/* Buttons */

div.stButton > button {
    border-radius: 10px;
    min-height: 45px;
    font-weight: 600;
}


/* Disclaimer */

.disclaimer {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(255, 170, 0, 0.5);
    margin-top: 25px;
    font-size: 14px;
}


/* Footer */

.app-footer {
    text-align: center;
    margin-top: 45px;
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


# =========================================================
# NAVIGATION FUNCTION
# =========================================================

def navigate(page):
    st.session_state.page = page
    st.session_state.prediction = None
    st.session_state.confidence = None
    st.rerun()


# =========================================================
# TOP NAVIGATION
# =========================================================

st.markdown(
    '<div class="nav-title">❤️ Hypertension Prediction System</div>',
    unsafe_allow_html=True
)

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    if st.button(
        "HOME",
        use_container_width=True
    ):
        navigate("Home")

with nav2:
    if st.button(
        "PREDICTION",
        use_container_width=True
    ):
        navigate("Prediction")

with nav3:
    if st.button(
        "MODEL",
        use_container_width=True
    ):
        navigate("Model")

with nav4:
    if st.button(
        "ABOUT",
        use_container_width=True
    ):
        navigate("About")

st.divider()


# =========================================================
# HOME PAGE
# =========================================================

if st.session_state.page == "Home":

    st.markdown(
        '<div class="hero-title">'
        'Hypertension Prediction System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Machine learning-based prediction of hypertension status'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="card">
            <div class="card-title">What does this system do?</div>
            <div class="card-text">
            This system uses selected health and lifestyle information
            to predict hypertension status using a Random Forest
            machine learning model.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">
            <div class="card-title">How does it work?</div>
            <div class="card-text">
            Enter the required information, submit the form, and the
            trained model will generate a predicted hypertension status
            together with its model confidence.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset", "NHANES 2017–2018")

    with col2:
        st.metric("Predictors", "7")

    with col3:
        st.metric("Algorithm", "Random Forest")

    st.divider()

    st.subheader("Ready to make a prediction?")

    if st.button(
        "START PREDICTION →",
        use_container_width=True
    ):
        navigate("Prediction")

    st.markdown("""
    <div class="disclaimer">
    <b>Disclaimer:</b><br>
    This research doesn't replace professional medical diagnosis.
    Please visit a professional doctor for further examination.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PREDICTION PAGE
# =========================================================

elif st.session_state.page == "Prediction":

    st.title("Hypertension Prediction")

    st.write(
        "Provide the required information below."
    )

    st.divider()

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

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
    # ACTION BUTTONS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        predict = st.button(
            "PREDICT STATUS",
            use_container_width=True
        )

    with col2:

        new_prediction = st.button(
            "CLEAR / NEW PREDICTION",
            use_container_width=True
        )

    # -----------------------------------------------------
    # CLEAR
    # -----------------------------------------------------

    if new_prediction:

        st.session_state.prediction = None
        st.session_state.confidence = None

        st.rerun()

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    if predict:

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

    # -----------------------------------------------------
    # DISPLAY RESULT
    # -----------------------------------------------------

    if st.session_state.prediction is not None:

        st.divider()

        st.subheader("Prediction Result")

        if st.session_state.prediction == 1:

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

        st.metric(
            "Model Confidence",
            f"{st.session_state.confidence:.2f}%"
        )

        st.divider()

        st.subheader("Submitted Information")

        result = pd.DataFrame({
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
        })

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("""
        <div class="disclaimer">
        <b>Disclaimer:</b><br>
        This research doesn't replace professional medical diagnosis.
        Please visit a professional doctor for further examination.
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# MODEL PAGE
# =========================================================

elif st.session_state.page == "Model":

    st.title("Model Information")

    st.write(
        "Technical information about the machine learning model "
        "used by the system."
    )

    st.divider()

    st.subheader("Random Forest")

    st.write(
        "The system uses the Random Forest classification algorithm "
        "as its primary predictive model."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Training Records",
            "5,224"
        )

    with col2:
        st.metric(
            "Train/Test Split",
            "80 / 20"
        )

    with col3:
        st.metric(
            "Trees",
            "200"
        )

    st.divider()

    st.subheader("Predictor Variables")

    predictors = pd.DataFrame({
        "No.": [1, 2, 3, 4, 5, 6, 7],

        "Variable": [
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
        "machine learning application developed to demonstrate "
        "the application of predictive modelling to hypertension "
        "status."
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
        "The system was developed using the NHANES 2017–2018 "
        "dataset."
    )

    st.subheader("Technologies")

    technology = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "Scikit-learn",
            "Joblib",
            "Streamlit"
        ],

        "Application": [
            "Programming",
            "Data processing",
            "Machine learning",
            "Model loading",
            "Web interface"
        ]
    })

    st.dataframe(
        technology,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("""
    <div class="disclaimer">
    <b>Medical Disclaimer</b><br><br>
    This research doesn't replace professional medical diagnosis.
    Please visit a professional doctor for further examination.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="app-footer">
    Hypertension Prediction System • Random Forest • NHANES 2017–2018
    </div>
    """,
    unsafe_allow_html=True
)

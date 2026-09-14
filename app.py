import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hypertension Prediction System",
    page_icon="🫀",
    layout="centered"
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
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 900px;
}

.app-title {
    font-size: 28px;
    font-weight: 700;
    line-height: 1.2;
}

.app-subtitle {
    font-size: 14px;
    opacity: 0.65;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 15px;
}

.info-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 15px;
}

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

@st.cache_resource
def load_model():
    return joblib.load("hypertension_random_forest.pkl")


try:
    model = load_model()
except Exception:
    st.error("The prediction model could not be loaded.")
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
# NAVIGATION
# ============================================================

def go_to(page):
    st.session_state.page = page
    st.session_state.menu_open = False


# ============================================================
# TOP HEADER
# ============================================================

left, right = st.columns([8.5, 1.5])

with left:
    st.markdown(
        '<div class="app-title">🫀 Hypertension Prediction System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-subtitle">Machine Learning Research System</div>',
        unsafe_allow_html=True
    )

with right:
    st.write("")

    if st.button(
        "☰",
        key="navigation_button",
        use_container_width=True
    ):
        st.session_state.menu_open = not st.session_state.menu_open


# ============================================================
# NAVIGATION MENU
# ============================================================

if st.session_state.menu_open:

    st.markdown("### Navigation")

    nav1, nav2 = st.columns(2)

    with nav1:

        if st.button("🏠 Home", use_container_width=True):
            go_to("Home")
            st.rerun()

        if st.button("🧠 Prediction", use_container_width=True):
            go_to("Prediction")
            st.rerun()

    with nav2:

        if st.button("📊 Model Information", use_container_width=True):
            go_to("Model Information")
            st.rerun()

        if st.button("ℹ️ About System", use_container_width=True):
            go_to("About System")
            st.rerun()


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":

    st.write("")

    st.markdown(
        '<div class="section-title">Hypertension Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "A machine learning research system designed to classify "
        "hypertension status using selected health and lifestyle "
        "characteristics."
    )

    st.write("")

    if st.button(
        "🧠 Start Prediction",
        use_container_width=True,
        type="primary"
    ):
        go_to("Prediction")
        st.rerun()

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
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

    with col2:
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

    st.info(
        "This research system does not replace professional "
        "medical diagnosis."
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
        "Enter the required information below."
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

        submit = st.form_submit_button(
            "🔍 Predict Hypertension Status",
            use_container_width=True,
            type="primary"
        )


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    if submit:

        # Convert categorical answers to the same numerical
        # format used during model training.

        sex_value = 1 if sex == "Male" else 0

        cholesterol_value = (
            1 if cholesterol == "Yes" else 0
        )

        diabetes_value = (
            1 if diabetes == "Yes" else 0
        )

        smoking_value = (
            1 if smoking == "Yes" else 0
        )

        physical_activity_value = (
            1 if physical_activity == "Yes" else 0
        )

        # ----------------------------------------------------
        # INPUT DATA
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [[
                age,
                sex_value,
                bmi,
                cholesterol_value,
                diabetes_value,
                smoking_value,
                physical_activity_value
            ]],
            columns=[
                "Age",
                "Sex",
                "BMI",
                "High_Cholesterol",
                "Diabetes",
                "Smoking_History",
                "Physical_Activity"
            ]
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        try:

            prediction = model.predict(input_data)[0]

            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = None

            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(
                    input_data
                )[0]

                confidence = float(
                    max(probability) * 100
                )

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

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

            # Move to result page
            st.session_state.page = "Result"

            st.session_state.menu_open = False

            # IMPORTANT:
            # Rerun only AFTER the prediction has been saved.
            st.rerun()

        except Exception as error:

            st.error(
                "An error occurred while making the prediction."
            )

            st.exception(error)


# ============================================================
# RESULT PAGE
# ============================================================

elif st.session_state.page == "Result":

    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.write("")

    prediction = st.session_state.prediction

    confidence = st.session_state.confidence

    patient_data = st.session_state.patient_data


    # ========================================================
    # CHECK RESULT
    # ========================================================

    if prediction is None:

        st.warning(
            "No prediction is available. Please make a new prediction."

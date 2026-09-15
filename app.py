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

/* Hide Streamlit default elements */
#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* Main page spacing */
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}


/* ============================================================
   TOP HEADER
============================================================ */

.app-title {
    font-size: 1.35rem;
    font-weight: 750;
    line-height: 1.2;
    margin-top: 5px;
}

.app-subtitle {
    font-size: 0.8rem;
    opacity: 0.65;
    margin-top: 4px;
}


/* Navigation button */
.nav-button button {
    border-radius: 12px;
    font-size: 1.25rem;
    font-weight: 700;
    min-height: 42px;
    padding: 0;
}


/* ============================================================
   GENERAL BUTTONS
============================================================ */

div.stButton > button {
    border-radius: 11px;
    min-height: 44px;
    font-weight: 600;
}


/* ============================================================
   HERO
============================================================ */

.hero {
    padding: 38px 30px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        rgba(255,245,247,1),
        rgba(255,255,255,1)
    );
    border: 1px solid rgba(180,40,60,0.10);
    margin-top: 20px;
    margin-bottom: 25px;
}

.hero h1 {
    margin: 0;
    font-size: 2.35rem;
    line-height: 1.15;
}

.hero p {
    margin-top: 13px;
    font-size: 1.03rem;
    opacity: 0.72;
    max-width: 780px;
    line-height: 1.6;
}


/* ============================================================
   INFORMATION CARDS
============================================================ */

.info-card {
    border: 1px solid rgba(128,128,128,0.16);
    border-radius: 18px;
    padding: 22px;
    min-height: 155px;
    background: rgba(255,255,255,0.7);
    box-shadow: 0 6px 22px rgba(0,0,0,0.04);
}

.info-card h3 {
    margin-top: 0;
    font-size: 1.08rem;
}

.info-card p {
    opacity: 0.72;
    line-height: 1.55;
}


/* ============================================================
   SECTION TITLE
============================================================ */

.section-title {
    font-size: 1.6rem;
    font-weight: 750;
    margin-top: 15px;
    margin-bottom: 5px;
}


/* ============================================================
   NAVIGATION MENU
============================================================ */

.menu-box {
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 16px;
    padding: 14px;
    margin-top: 8px;
    margin-bottom: 20px;
    max-width: 450px;
    margin-left: auto;
    box-shadow: 0 8px 25px rgba(0,0,0,0.07);
}


/* ============================================================
   RESULT AREA
============================================================ */

.result-panel {
    padding: 30px;
    border-radius: 22px;
    border: 1px solid rgba(128,128,128,0.16);
    box-shadow: 0 8px 28px rgba(0,0,0,0.06);
    text-align: center;
    margin-top: 20px;
    margin-bottom: 25px;
}


/* ============================================================
   FOOTER
============================================================ */

.custom-footer {
    text-align: center;
    opacity: 0.55;
    font-size: 0.78rem;
    margin-top: 45px;
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


try:

    model = load_model()

except Exception as error:

    st.error("The prediction model could not be loaded.")

    st.write(
        "Make sure the file "
        "'hypertension_random_forest.pkl' "
        "is in the same folder as app.py."
    )

    st.code(str(error))

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

header_left, header_right = st.columns([10, 1])

with header_left:

    st.markdown(
        '<div class="app-title">'
        '🫀 Hypertension Prediction System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-subtitle">'
        'Machine Learning Research System'
        '</div>',
        unsafe_allow_html=True
    )


with header_right:

    st.markdown('<div class="nav-button">', unsafe_allow_html=True)

    if st.button(
        "☰",
        key="navigation_button",
        help="Open navigation",
        use_container_width=True
    ):

        st.session_state.menu_open = (
            not st.session_state.menu_open
        )

        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# NAVIGATION MENU
# ============================================================

if st.session_state.menu_open:

    st.markdown(
        '<div class="menu-box">',
        unsafe_allow_html=True
    )

    st.markdown("### Navigation")

    nav1, nav2 = st.columns(2)

    with nav1:

        if st.button(
            "🏠 Home",
            key="nav_home",
            use_container_width=True
        ):
            navigate("Home")

        if st.button(
            "🧠 Prediction",
            key="nav_prediction",
            use_container_width=True
        ):
            navigate("Prediction")

    with nav2:

        if st.button(
            "📊 Model Information",
            key="nav_model",
            use_container_width=True
        ):
            navigate("Model")

        if st.button(
            "ℹ️ About System",
            key="nav_about",
            use_container_width=True
        ):
            navigate("About")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "Home":

    st.markdown(
        """
        <div class="hero">

            <h1>
                Hypertension Prediction System
            </h1>

            <p>
                A machine learning research system designed to
                classify current hypertension status using selected
                health and lifestyle factors.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    card1, card2, card3 = st.columns(3)

    with card1:

        st.markdown(
            """
            <div class="info-card">

                <h3>🧠 Machine Learning</h3>

                <p>
                    Uses a Random Forest classification model
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

                <h3>📊 Seven Predictors</h3>

                <p>
                    Age, sex, BMI, high cholesterol, diabetes,
                    smoking history and physical activity.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with card3:

        st.markdown(
            """
            <div class="info-card">

                <h3>⚕️ Research Support</h3>

                <p>
                    Provides a machine learning classification
                    result for research and educational purposes.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")

    if st.button(
        "🧠 Start Prediction",
        key="home_prediction",
        use_container_width=True
    ):
        navigate("Prediction")


# ============================================================
# PREDICTION PAGE
# ============================================================

elif st.session_state.page == "Prediction":

    st.markdown(
        '<div class="section-title">Hypertension Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the required health and lifestyle information "
        "to obtain the model's predicted hypertension status."
    )

    st.write("")


    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Hypertension Status",
        key="predict_button",
        use_container_width=True
    ):

        try:

            # IMPORTANT:
            # These are the original feature names used by
            # the trained model.

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


            # Make prediction
            prediction = int(
                model.predict(input_data)[0]
            )


            # Get confidence if model supports probabilities
            confidence = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                confidence = float(
                    max(probabilities) * 100
                )


            # Save results
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


            # Go to result page
            st.session_state.page = "Result"

            st.rerun()


        except Exception as error:

            st.error(
                "The prediction could not be completed."
            )

            st.write(
                "Please check that the model file and its "
                "trained feature structure are correct."
            )

            st.code(str(error))


# ============================================================
# RESULT PAGE
# ============================================================

elif st.session_state.page == "Result":

    prediction = st.session_state.prediction
    confidence = st.session_state.confidence


    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.write("")


    if prediction is None:

        st.warning(
            "No prediction is available yet."
        )

        if st.button(
            "🧠 Make a Prediction",
            use_container_width=True
        ):
            navigate("Prediction")


    else:

        # ----------------------------------------------------
        # HYPERTENSION RESULT
        # ----------------------------------------------------

        if prediction == 1:

            st.markdown(
                '<div class="result-panel">',
                unsafe_allow_html=True
            )

            st.write("⚠️")

            st.error("HYPERTENSION")

            st.subheader(
                "The model predicts a positive hypertension status."
            )

            if confidence is not None:

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            st.warning(
                "This result should not be considered a medical "
                "diagnosis. Please consult a qualified healthcare "
                "professional for proper evaluation."
            )


        # ----------------------------------------------------
        # NON-HYPERTENSION RESULT
        # ----------------------------------------------------

        else:

            st.markdown(
                '<div class="result-panel">',
                unsafe_allow_html=True
            )

            st.write("✓")

            st.success("NON-HYPERTENSION")

            st.subheader(
                "The model predicts a negative hypertension status."
            )

            if confidence is not None:

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            st.info(
                "This result does not rule out hypertension and "
                "does not replace a professional medical assessment."
            )


        # ----------------------------------------------------
        # NEW PREDICTION
        # ----------------------------------------------------

        st.write("")

        if st.button(
            "🔄 New Prediction",
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

    st.markdown(
        '<div class="section-title">Model Information</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The system uses a Random Forest classification model "
        "trained with NHANES 2017–2018 data."
    )

    st.write("")


    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Accuracy",
            "71.29%"
        )

    with metric2:

        st.metric(
            "Precision",
            "60.91%"
        )

    with metric3:

        st.metric(
            "Recall",
            "54.03%"
        )


    metric4, metric5 = st.columns(2)

    with metric4:

        st.metric(
            "F1-Score",
            "57.26%"
        )

    with metric5:

        st.metric(
            "ROC-AUC",
            "76.57%"
        )


    st.write("")

    st.subheader("Predictors Used")

    predictors = pd.DataFrame({
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


# ============================================================
# ABOUT PAGE
# ============================================================

elif st.session_state.page == "About":

    st.markdown(
        '<div class="section-title">About the System</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The Hypertension Prediction System is a research and "
        "educational software application developed to demonstrate "
        "the use of machine learning for hypertension status "
        "classification."
    )

    st.write(
        "The system uses selected demographic, health and lifestyle "
        "information as input to a Random Forest classification model."
    )

    st.write("")

    st.subheader("Important Disclaimer")

    st.warning(
        "This research doesn't replace professional medical "
        "diagnosis. Please visit a professional doctor for "
        "further examination."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="custom-footer">
        © 2026 Hypertension Prediction System |
        Machine Learning Research Project
    </div>
    """,
    unsafe_allow_html=True
)

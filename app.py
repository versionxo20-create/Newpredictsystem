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

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}


/* ------------------------------------------------------------
   TOP HEADER
------------------------------------------------------------ */

.app-title {
    font-size: 1.35rem;
    font-weight: 750;
    line-height: 1.2;
}

.app-subtitle {
    font-size: 0.8rem;
    opacity: 0.65;
    margin-top: 4px;
}


/* ------------------------------------------------------------
   BUTTONS
------------------------------------------------------------ */

div.stButton > button {
    border-radius: 11px;
    min-height: 42px;
    font-weight: 600;
}


/* ------------------------------------------------------------
   HERO SECTION
------------------------------------------------------------ */

.hero {
    padding: 38px 30px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        rgba(255,245,247,1),
        rgba(255,255,255,1)
    );
    border: 1px solid rgba(180,40,60,0.10);
    margin-top: 15px;
    margin-bottom: 25px;
}

.hero h1 {
    margin: 0;
    font-size: 2.4rem;
    line-height: 1.15;
}

.hero p {
    margin-top

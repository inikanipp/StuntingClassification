import streamlit as st
import joblib
import numpy as np
from scipy.stats import boxcox
from datetime import date

st.set_page_config(page_title="Stunting Prediction", layout="wide")

# ===== Background (optional UI only) =====
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        background-color: #f8f9fa !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== Load model =====
model = joblib.load("pages/model2.pkl")
minmax = joblib.load("pages/minmax.pkl")

# ===== Lambda Box-Cox values (FROM TRAINING – DO NOT CHANGE) =====
LBirthWeight = -1.3288
LBirthHeight = 4.5616
LWeight = 1.0253
LHeight = 2.4460

st.markdown(
    "<h1 style='text-align:center;'>🧒 Stunting Prediction System (KNN)</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# Two-column layout
left_col, right_col = st.columns([2, 1])

# ================= LEFT: INPUTS =================
with left_col:
    st.subheader("📋 Child Information")

    birth_weight = st.number_input("Birth Weight (kg)", min_value=0.01, step=0.1)
    birth_height = st.number_input("Birth Height (cm)", min_value=0.01, step=1.0)
    weight = st.number_input("Current Weight (kg)", min_value=0.01, step=0.1)
    height = st.number_input("Current Height (cm)", min_value=0.01, step=1.0)

    age_years = st.number_input("Age (Years)", min_value=0.0, step=0.1)

    predict_btn = st.button("🔍 Predict Status")

# ================= RIGHT: OUTPUT =================
with right_col:
    st.subheader("📊 Prediction Result")

    st.info(
        "This system uses the K-Nearest Neighbors (KNN) algorithm. "
        "User inputs are transformed using Box-Cox to match the training data."
    )

    if predict_btn:
        try:
            # ===== Apply Box-Cox (same as training) =====
            birth_weight_bc = boxcox([birth_weight], lmbda=LBirthWeight)[0]
            birth_height_bc = boxcox([birth_height], lmbda=LBirthHeight)[0]
            weight_bc = boxcox([weight], lmbda=LWeight)[0]
            height_bc = boxcox([height], lmbda=LHeight)[0]

            # ===== Combine features (ORDER MUST MATCH TRAINING) =====
            input_data = np.array([[
                birth_weight_bc,
                birth_height_bc,
                weight_bc,
                height_bc,
                age_years
            ]])     
            

            # ======================= transform normalized ===================================
            input_data = minmax.transform(input_data)
            # ======================= predict ===================================
            prediction = model.predict(input_data)

            label_map = {
                0: "Very Short",
                1: "Short"
            }

            st.success(f"📌 Predicted TB/U Status: **{label_map[prediction[0]]}**")

            st.caption(
                "Note: Box-Cox transformation is applied to ensure consistency "
                "between training and prediction data."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
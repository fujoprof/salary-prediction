import numpy as np
import pandas as pd
import streamlit as st
import pickle


model = pickle.load(open("salary_prediction_model","rb"))

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="💼 Salary Prediction App", layout="centered")

st.title("💼 Salary Prediction Based on Experience")
st.markdown("Enter the **years of experience** below to predict the expected salary.")

# Input field
years_exp = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.1)


# Prediction button
if st.button("🔮 Predict Salary"):
    # Prepare input (must be 2D for sklearn)
    input_data = np.array(years_exp).reshape(-1,1)
    prediction = model.predict(input_data)

    st.success(f"Predicted Salary: **${prediction[0][0]:,.2f}**")
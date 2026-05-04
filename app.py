import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

# Title
st.title("📊 Student Exam Score Predictor")

st.header("Enter Study Details:")

# Input
study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, step=0.5)

# Prediction
if st.button("Predict Score"):
    input_data = np.array([[study_hours]])
    prediction = model.predict(input_data)

    st.success(f"📈 Predicted Exam Score: {prediction[0]:.2f}")
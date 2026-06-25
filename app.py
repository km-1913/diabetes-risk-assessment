import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("best_diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.title("Diabetes Risk Assessment System")

st.write("Enter patient details below:")

# Input Fields
preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Prediction Button
if st.button("Predict Diabetes Risk"):

    # Input array
    input_data = np.array([[
        preg,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Display Result
    if prediction == 1:
        st.error("High Risk: Patient is likely to have diabetes.")
    else:
        st.success("Low Risk: Patient is unlikely to have diabetes.")

    # Show prediction details
    st.write(f"Prediction Output: {prediction}")
    st.write(f"Diabetes Probability: {probability:.2f}")
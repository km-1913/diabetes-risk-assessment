import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("best_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Diabetes Risk Assessment System")

st.write("Enter patient details below:")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1)

if st.button("Predict Diabetes Risk"):

    patient_data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]]

    patient_data_scaled = scaler.transform(patient_data)

    prediction = model.predict(patient_data_scaled)

    if prediction[0] == 1:
        st.error("High Risk: Patient may have diabetes.")

        st.write("Recommendations:")
        st.write("- Consult a doctor")
        st.write("- Exercise regularly")
        st.write("- Reduce sugar intake")

    else:
        st.success("Low Risk: Patient is unlikely to have diabetes.")

        st.write("Recommendations:")
        st.write("- Maintain healthy lifestyle")
        st.write("- Continue regular exercise")
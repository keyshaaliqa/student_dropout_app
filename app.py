import sys
# If Streamlit is not installed, run in your shell:
#   python -m pip install streamlit
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model dan scaler
model = joblib.load('dropout_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa",
    page_icon="🎓"
)

st.title("🎓 Prediksi Dropout Mahasiswa")
st.write("Masukkan data mahasiswa untuk memprediksi status akademik.")

age = st.number_input(
    "Age at Enrollment",
    min_value=17,
    max_value=70,
    value=20
)

admission_grade = st.number_input(
    "Admission Grade",
    min_value=0.0,
    max_value=200.0,
    value=120.0
)

sem1_grade = st.number_input(
    "1st Semester Grade",
    min_value=0.0,
    max_value=20.0,
    value=10.0
)

sem2_grade = st.number_input(
    "2nd Semester Grade",
    min_value=0.0,
    max_value=20.0,
    value=10.0
)

tuition = st.selectbox(
    "Tuition Fees Up To Date",
    [0, 1]
)

scholarship = st.selectbox(
    "Scholarship Holder",
    [0, 1]
)

if st.button("Prediksi"):

    data = np.array([[
        age,
        admission_grade,
        sem1_grade,
        sem2_grade,
        tuition,
        scholarship
    ]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    label_map = {
        0: "Dropout",
        1: "Enrolled",
        2: "Graduate"
    }

    st.success(
        f"Hasil Prediksi: {label_map[prediction[0]]}"
    )

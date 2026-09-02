import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================
# LOAD MODEL & SCALER
# =========================
model = joblib.load("dropout_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa",
    page_icon="🎓"
)

st.title("🎓 Prediksi Dropout Mahasiswa")
st.write("Masukkan data mahasiswa untuk memprediksi status akademik.")

# =========================
# INPUT DATA
# =========================

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

# =========================
# PREDICTION
# =========================

if st.button("Prediksi"):

    # Gunakan DataFrame agar nama dan urutan fitur jelas
    data = pd.DataFrame([{
        "Age at Enrollment": age,
        "Admission Grade": admission_grade,
        "1st Semester Grade": sem1_grade,
        "2nd Semester Grade": sem2_grade,
        "Tuition Fees Up To Date": tuition,
        "Scholarship Holder": scholarship
    }])

    try:

        # Cek jumlah fitur scaler
        expected_features = scaler.n_features_in_

        if data.shape[1] != expected_features:
            st.error(
                f"Jumlah fitur tidak sesuai. "
                f"Scaler membutuhkan {expected_features} fitur, "
                f"tetapi aplikasi memberikan {data.shape[1]} fitur."
            )

        else:

            # Scaling
            data_scaled = scaler.transform(data)

            # Prediction
            prediction = model.predict(data_scaled)

            label_map = {
                0: "Dropout",
                1: "Graduate"
            }

            result = label_map.get(
                int(prediction[0]),
                str(prediction[0])
            )

            st.success(f"🎓 Hasil Prediksi: {result}")

    except Exception as e:

        st.error(
            f"Terjadi error saat melakukan prediksi: {str(e)}"
        )
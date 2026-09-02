import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
model_data = joblib.load("model.pkl")

model = model_data["model"]
preprocessor = model_data["preprocessor"]
label_encoder = model_data["label_encoder"]

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

# =========================
# JUDUL APLIKASI
# =========================
st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.write(
    "Aplikasi ini digunakan untuk memprediksi apakah mahasiswa "
    "berpotensi **Dropout** atau **Graduate** berdasarkan data akademik "
    "dan karakteristik mahasiswa."
)

st.divider()

# =========================
# INPUT DATA
# =========================
age = st.number_input(
    "Age at Enrollment",
    min_value=15,
    max_value=100,
    value=18
)

admission_grade = st.number_input(
    "Admission Grade",
    min_value=0.0,
    max_value=200.0,
    value=120.0
)

grade_1 = st.number_input(
    "1st Semester Grade",
    min_value=0.0,
    max_value=20.0,
    value=10.0
)

grade_2 = st.number_input(
    "2nd Semester Grade",
    min_value=0.0,
    max_value=20.0,
    value=10.0
)

tuition = st.selectbox(
    "Tuition Fees Up To Date",
    [0, 1],
    index=1
)

scholarship = st.selectbox(
    "Scholarship Holder",
    [0, 1],
    index=1
)

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi", use_container_width=True):

    input_data = pd.DataFrame([{
        "Age at enrollment": age,
        "Admission grade": admission_grade,
        "Curricular units 1st sem (grade)": grade_1,
        "Curricular units 2nd sem (grade)": grade_2,
        "Tuition fees up to date": tuition,
        "Scholarship holder": scholarship
    }])

    try:
        X_input = preprocessor.transform(input_data)

        prediction = model.predict(X_input)[0]
        result = label_encoder.inverse_transform([prediction])[0]

        if result == "Dropout":
            st.error("⚠️ Prediksi: **DROPOUT**")
        else:
            st.success("🎓 Prediksi: **GRADUATE**")

    except Exception as e:
        st.error(f"❌ Terjadi error saat melakukan prediksi: {e}")

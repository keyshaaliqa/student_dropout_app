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

st.subheader("📋 Input Data Mahasiswa")

marital_status = st.number_input("Marital status", min_value=0, value=1)
application_mode = st.number_input("Application mode", min_value=0, value=1)
application_order = st.number_input("Application order", min_value=0, value=1)
course = st.number_input("Course", min_value=0, value=1)
daytime_evening = st.number_input(
    "Daytime/evening attendance",
    min_value=0,
    value=1
)

previous_qualification = st.number_input(
    "Previous qualification",
    min_value=0,
    value=1
)

previous_qualification_grade = st.number_input(
    "Previous qualification (grade)",
    min_value=0.0,
    value=120.0
)

nationality = st.number_input(
    "Nationality",
    min_value=0,
    value=1
)

mother_qualification = st.number_input(
    "Mother's qualification",
    min_value=0,
    value=1
)

father_qualification = st.number_input(
    "Father's qualification",
    min_value=0,
    value=1
)

mother_occupation = st.number_input(
    "Mother's occupation",
    min_value=0,
    value=1
)

father_occupation = st.number_input(
    "Father's occupation",
    min_value=0,
    value=1
)

admission_grade = st.number_input(
    "Admission grade",
    min_value=0.0,
    value=120.0
)

displaced = st.number_input(
    "Displaced",
    min_value=0,
    value=0
)

educational_special_needs = st.number_input(
    "Educational special needs",
    min_value=0,
    value=0
)

debtor = st.number_input(
    "Debtor",
    min_value=0,
    value=0
)

tuition_fees = st.number_input(
    "Tuition fees up to date",
    min_value=0,
    value=1
)

gender = st.number_input(
    "Gender",
    min_value=0,
    value=1
)

scholarship_holder = st.number_input(
    "Scholarship holder",
    min_value=0,
    value=0
)

age = st.number_input(
    "Age at enrollment",
    min_value=15,
    max_value=100,
    value=18
)

international = st.number_input(
    "International",
    min_value=0,
    value=0
)

# Semester 1
curricular_1_credited = st.number_input(
    "Curricular units 1st sem (credited)",
    min_value=0,
    value=0
)

curricular_1_enrolled = st.number_input(
    "Curricular units 1st sem (enrolled)",
    min_value=0,
    value=0
)

curricular_1_evaluations = st.number_input(
    "Curricular units 1st sem (evaluations)",
    min_value=0,
    value=0
)

curricular_1_approved = st.number_input(
    "Curricular units 1st sem (approved)",
    min_value=0,
    value=0
)

curricular_1_grade = st.number_input(
    "Curricular units 1st sem (grade)",
    min_value=0.0,
    value=0.0
)

curricular_1_without_evaluations = st.number_input(
    "Curricular units 1st sem (without evaluations)",
    min_value=0,
    value=0
)

# Semester 2
curricular_2_credited = st.number_input(
    "Curricular units 2nd sem (credited)",
    min_value=0,
    value=0
)

curricular_2_enrolled = st.number_input(
    "Curricular units 2nd sem (enrolled)",
    min_value=0,
    value=0
)

curricular_2_evaluations = st.number_input(
    "Curricular units 2nd sem (evaluations)",
    min_value=0,
    value=0
)

curricular_2_approved = st.number_input(
    "Curricular units 2nd sem (approved)",
    min_value=0,
    value=0
)

curricular_2_grade = st.number_input(
    "Curricular units 2nd sem (grade)",
    min_value=0.0,
    value=0.0
)

curricular_2_without_evaluations = st.number_input(
    "Curricular units 2nd sem (without evaluations)",
    min_value=0,
    value=0
)

unemployment_rate = st.number_input(
    "Unemployment rate",
    min_value=0.0,
    value=10.0
)

inflation_rate = st.number_input(
    "Inflation rate",
    value=1.0
)

gdp = st.number_input(
    "GDP",
    value=1.0
)

# =========================
# PREDIKSI
# =========================
if st.button(
    "🔍 Prediksi", 
    use_container_width=True, 
    key="prediksi_button"
):
    input_data = pd.DataFrame([{
    "Marital status": marital_status,
    "Application mode": application_mode,
    "Application order": application_order,
    "Course": course,
    "Daytime/evening attendance\t": daytime_evening,
    "Previous qualification": previous_qualification,
    "Previous qualification (grade)": previous_qualification_grade,
    "Nacionality": nationality,
    "Mother's qualification": mother_qualification,
    "Father's qualification": father_qualification,
    "Mother's occupation": mother_occupation,
    "Father's occupation": father_occupation,
    "Admission grade": admission_grade,
    "Displaced": displaced,
    "Educational special needs": educational_special_needs,
    "Debtor": debtor,
    "Tuition fees up to date": tuition_fees,
    "Gender": gender,
    "Scholarship holder": scholarship_holder,
    "Age at enrollment": age,
    "International": international,
    "Curricular units 1st sem (credited)": curricular_1_credited,
    "Curricular units 1st sem (enrolled)": curricular_1_enrolled,
    "Curricular units 1st sem (evaluations)": curricular_1_evaluations,
    "Curricular units 1st sem (approved)": curricular_1_approved,
    "Curricular units 1st sem (grade)": curricular_1_grade,
    "Curricular units 1st sem (without evaluations)": curricular_1_without_evaluations,
    "Curricular units 2nd sem (credited)": curricular_2_credited,
    "Curricular units 2nd sem (enrolled)": curricular_2_enrolled,
    "Curricular units 2nd sem (evaluations)": curricular_2_evaluations,
    "Curricular units 2nd sem (approved)": curricular_2_approved,
    "Curricular units 2nd sem (grade)": curricular_2_grade,
    "Curricular units 2nd sem (without evaluations)": curricular_2_without_evaluations,
    "Unemployment rate": unemployment_rate,
    "Inflation rate": inflation_rate,
    "GDP": gdp
}])

if st.button(
    "🔍 Prediksi", 
    use_container_width=True, 
    key="prediksi_button"
):
    try:
        X_input = preprocessor.transform(input_data)

        pprediction = model.predict(data_scaled)

        result = label_encoder.inverse_transform(
            prediction.astype(int)
        )[0]

st.success(f"🎓 Hasil Prediksi: {result}")

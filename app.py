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

marital_status = st.number_input("1. Marital status", min_value=0, value=1)
application_mode = st.number_input("2. Application mode", min_value=0, value=1)
application_order = st.number_input("3. Application order", min_value=0, value=1)
course = st.number_input("4. Course", min_value=0, value=1)
daytime_evening = st.number_input(
    "5. Daytime/evening attendance",
    min_value=0,
    value=1
)

previous_qualification = st.number_input(
    "6. Previous qualification",
    min_value=0,
    value=1
)

previous_qualification_grade = st.number_input(
    "7. Previous qualification (grade)",
    min_value=0.0,
    value=120.0
)

nationality = st.number_input(
    "8. Nacionality",
    min_value=0,
    value=1
)

mother_qualification = st.number_input(
    "9. Mother's qualification",
    min_value=0,
    value=1
)

father_qualification = st.number_input(
    "10. Father's qualification",
    min_value=0,
    value=1
)

mother_occupation = st.number_input(
    "11. Mother's occupation",
    min_value=0,
    value=1
)

father_occupation = st.number_input(
    "12. Father's occupation",
    min_value=0,
    value=1
)

admission_grade = st.number_input(
    "13. Admission grade",
    min_value=0.0,
    value=120.0
)

displaced = st.number_input(
    "14. Displaced",
    min_value=0,
    value=0
)

educational_special_needs = st.number_input(
    "15. Educational special needs",
    min_value=0,
    value=0
)

debtor = st.number_input(
    "16. Debtor",
    min_value=0,
    value=0
)

tuition_fees = st.number_input(
    "17. Tuition fees up to date",
    min_value=0,
    value=1
)

gender = st.number_input(
    "18. Gender",
    min_value=0,
    value=1
)

scholarship_holder = st.number_input(
    "19. Scholarship holder",
    min_value=0,
    value=0
)

age = st.number_input(
    "20. Age at enrollment",
    min_value=15,
    max_value=100,
    value=18
)

international = st.number_input(
    "21. International",
    min_value=0,
    value=0
)

# Semester 1
curricular_1_credited = st.number_input(
    "22. Curricular units 1st sem (credited)",
    min_value=0,
    value=0
)

curricular_1_enrolled = st.number_input(
    "23. Curricular units 1st sem (enrolled)",
    min_value=0,
    value=0
)

curricular_1_evaluations = st.number_input(
    "24. Curricular units 1st sem (evaluations)",
    min_value=0,
    value=0
)

curricular_1_approved = st.number_input(
    "25. Curricular units 1st sem (approved)",
    min_value=0,
    value=0
)

curricular_1_grade = st.number_input(
    "26. Curricular units 1st sem (grade)",
    min_value=0.0,
    value=0.0
)

curricular_1_without_evaluations = st.number_input(
    "27. Curricular units 1st sem (without evaluations)",
    min_value=0,
    value=0
)

# Semester 2
curricular_2_credited = st.number_input(
    "28. Curricular units 2nd sem (credited)",
    min_value=0,
    value=0
)

curricular_2_enrolled = st.number_input(
    "29. Curricular units 2nd sem (enrolled)",
    min_value=0,
    value=0
)

curricular_2_evaluations = st.number_input(
    "30. Curricular units 2nd sem (evaluations)",
    min_value=0,
    value=0
)

curricular_2_approved = st.number_input(
    "31. Curricular units 2nd sem (approved)",
    min_value=0,
    value=0
)

curricular_2_grade = st.number_input(
    "32. Curricular units 2nd sem (grade)",
    min_value=0.0,
    value=0.0
)

curricular_2_without_evaluations = st.number_input(
    "33. Curricular units 2nd sem (without evaluations)",
    min_value=0,
    value=0
)

unemployment_rate = st.number_input(
    "34. Unemployment rate",
    min_value=0.0,
    value=10.0
)

inflation_rate = st.number_input(
    "35. Inflation rate",
    value=1.0
)

gdp = st.number_input(
    "36. GDP",
    value=1.0
)

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi", use_container_width=True):

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

if st.button("🔍 Prediksi", use_container_width=True):
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

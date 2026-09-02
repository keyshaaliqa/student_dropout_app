import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
model_data = joblib.load("model.pkl")

model = model_data["model"]
scaler = model_data["scaler"]


# =========================
# INPUT
# =========================
age = st.number_input("Age at Enrollment", min_value=15, max_value=100, value=18)
admission_grade = st.number_input("Admission Grade", value=120.0)
grade_1 = st.number_input("1st Semester Grade", value=10.0)
grade_2 = st.number_input("2nd Semester Grade", value=10.0)

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
# PREDICTION
# =========================
if st.button("Prediksi"):

    input_data = pd.DataFrame({
        "Age at Enrollment": [age],
        "Admission Grade": [admission_grade],
        "1st Semester Grade": [grade_1],
        "2nd Semester Grade": [grade_2],
        "Tuition Fees Up To Date": [tuition],
        "Scholarship Holder": [scholarship]
    })

    try:
        X_input = scaler.transform(input_data)
        prediction = model.predict(X_input)[0]

        if prediction == 0:
            st.error("⚠️ Prediksi: Dropout")
        else:
            st.success("🎓 Prediksi: Graduate")

    except Exception as e:
        st.error(f"❌ Terjadi error saat melakukan prediksi: {e}")

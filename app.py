import streamlit as st
import joblib
import pandas as pd

# =========================
# LOAD MODEL & SCALER
# =========================

model = joblib.load("dropout_model.pkl")
scaler = joblib.load("scaler_6_features.pkl")

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

semester1_grade = st.number_input(
    "1st Semester Grade",
    min_value=0.0,
    max_value=20.0,
    value=10.0
)

semester2_grade = st.number_input(
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

    try:

        # Nama fitur harus SAMA PERSIS dengan saat training
        feature_names = [
            "Age at enrollment",
            "Admission grade",
            "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (grade)",
            "Tuition fees up to date",
            "Scholarship holder"
        ]

        # Membuat DataFrame input
        data = pd.DataFrame([[
            age,
            admission_grade,
            semester1_grade,
            semester2_grade,
            tuition,
            scholarship
        ]], columns=feature_names)

        # =========================
        # CEK FITUR SCALER
        # =========================

        if hasattr(scaler, "feature_names_in_"):

            scaler_features = list(scaler.feature_names_in_)

            if scaler_features != feature_names:

                st.error(
                    "❌ Nama atau urutan fitur pada scaler tidak sesuai."
                )

                st.write("Fitur aplikasi:")
                st.write(feature_names)

                st.write("Fitur scaler:")
                st.write(scaler_features)

                st.stop()

        # =========================
        # SCALING
        # =========================

        data_scaled = scaler.transform(data)

        # =========================
        # PREDICTION
        # =========================

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
            f"❌ Terjadi error saat melakukan prediksi: {str(e)}"
        )

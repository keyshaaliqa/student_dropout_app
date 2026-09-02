import streamlit as st
import joblib
import pandas as pd


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa",
    page_icon="🎓"
)


# =========================
# LOAD MODEL & SCALER
# =========================

model = joblib.load(
    "dropout_model_6_features.pkl"
)

scaler = joblib.load(
    "scaler_6_features.pkl"
)


# =========================
# TITLE
# =========================

st.title("🎓 Prediksi Dropout Mahasiswa")

st.write(
    "Masukkan data mahasiswa untuk memprediksi "
    "status akademik."
)


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

    try:

        # DataFrame dengan nama fitur
        # yang SAMA dengan saat training

        data = pd.DataFrame([{
            "Age at enrollment": age,
            "Admission grade": admission_grade,
            "Curricular units 1st sem (grade)": sem1_grade,
            "Curricular units 2nd sem (grade)": sem2_grade,
            "Tuition fees up to date": tuition,
            "Scholarship holder": scholarship
        }])

        # =========================
        # VALIDASI
        # =========================

        if data.shape[1] != model.n_features_in_:

            st.error(
                f"Jumlah fitur tidak sesuai. "
                f"Model membutuhkan "
                f"{model.n_features_in_} fitur, "
                f"tetapi diberikan "
                f"{data.shape[1]} fitur."
            )

            st.stop()


        # =========================
        # SCALING
        # =========================

        data_scaled = scaler.transform(data)


        # =========================
        # PREDICTION
        # =========================

        prediction = model.predict(data_scaled)


        # =========================
        # RESULT
        # =========================

        label_map = {
            0: "Dropout",
            1: "Graduate"
        }

        result = label_map.get(
            int(prediction[0]),
            "Unknown"
        )


        if result == "Dropout":

            st.error(
                f"📌 Hasil Prediksi: **{result}**"
            )

        else:

            st.success(
                f"🎓 Hasil Prediksi: **{result}**"
            )


    except Exception as e:

        st.error(
            f"❌ Terjadi error saat melakukan "
            f"prediksi: {str(e)}"
        )

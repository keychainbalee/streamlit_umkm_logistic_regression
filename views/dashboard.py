import pickle
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
import numpy as np

# =====================================
# CONSTANTS
# =====================================

EDUCATION_MAP = {
    1: "SD",
    2: "SMP",
    3: "SMA",
    4: "Sarjana",
    5: "Magister"
}

GENDER_MAP = {
    0: "Perempuan",
    1: "Laki-laki"
}


# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    with open("notebook_state (1).pkl", "rb") as file:
        return pickle.load(file)
    
def get_top_feature(data):

    importance_df = data["feature_importance_df"]

    top_feature = importance_df.iloc[0]

    return top_feature
# =====================================
# CSS
# =====================================

def load_css():
    with open("assets/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


# =====================================
# DATASET
# =====================================

def render_dataset_filter():

    st.subheader("Dataset")

    return st.selectbox(
        "Pilih Dataset",
        ["UMKM RegLog"],
        label_visibility="collapsed"
    )


# =====================================
# FEATURE IMPORTANCE
# =====================================

def render_feature_importance(data):

    top_feature = get_top_feature(data)

    feature_name = (
        str(top_feature["Feature"])
        .replace("_", " ")
        .title()
    )

    st.markdown("""
    <style>
    .feature-box {
        background: linear-gradient(135deg, #1e3a8a, #172554);
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        border: 1px solid #2563eb;
        margin-bottom: 20px;
    }
    .feature-title {
        color: #bfdbfe;
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .feature-value {
        color: white;
        font-size: 56px;
        font-weight: 800;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="feature-box">
            <div class="feature-title">
                Faktor Paling Berpengaruh
            </div>
            <div class="feature-value">
                {feature_name}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================
# INSIGHT
# =====================================

def render_insight_cards(data):

    best_age = data["age_success_mapping"][1].idxmax()

    best_education = EDUCATION_MAP.get(
        data["education_success_mapping"][1].idxmax()
    )

    best_gender = GENDER_MAP.get(
        data["gender_success_mapping"][1].idxmax()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Umur Paling Sukses",
            best_age
        )

    with col2:
        st.metric(
            "Pendidikan Paling Sukses",
            best_education
        )

    with col3:
        st.metric(
            "Gender Dominan",
            best_gender
        )
        
def render_policy_recommendation(data):

    top_features = data["top_3_features"]

    st.subheader("Rekomendasi Kebijakan")

    st.caption(
        "Faktor yang memiliki pengaruh terbesar terhadap keberhasilan UMKM berdasarkan nilai Odds Ratio dari model Logistic Regression."
    )

    cols = st.columns(3)

    for col, (_, row) in zip(
        cols,
        top_features.iterrows()
    ):

        feature = (
            str(row["Feature"])
            .replace("_", " ")
            .title()
        )

        odds = float(row["Odds_Ratio"])

        description = (
            f"UMKM yang memiliki "
            f"{feature.lower()} yang baik "
            f"berpotensi memiliki peluang sukses "
            f"{odds:.2f} kali lebih besar."
        )

        with col:

            st.metric(
                label=feature,
                value=f"{odds:.2f}x"
            )

            st.caption(description)

    st.success(
        """
        Berdasarkan hasil analisis Logistic Regression,
        peningkatan faktor-faktor di atas dapat menjadi
        fokus utama dalam strategi pengembangan UMKM.
        """
    )

    st.info(
        """
        Insight ini dihasilkan secara otomatis dari model Logistic Regression
        dan akan diperbarui ketika dataset berubah.
        """
    )
    
# =====================================
# PAGE
# =====================================

def show_dashboard():

    load_css()

    data = load_data()

    st.title("Profil Kesuksesan UMKM")

    st.caption(
        "Analisis faktor yang memengaruhi keberhasilan UMKM menggunakan Logistic Regression"
    )

    st.divider()

    render_dataset_filter()

    st.divider()

    render_feature_importance(data)

    st.divider()

    render_insight_cards(data)

    st.divider()

    render_policy_recommendation(data)
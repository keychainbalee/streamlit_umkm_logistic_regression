import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():

    pkl_path = (
        Path(__file__).parent.parent
        / "notebook_state (1).pkl"
    )

    with open(pkl_path, "rb") as file:
        return pickle.load(file)


# =====================================
# LABEL MAPPING
# =====================================

FEATURE_LABELS = {

    "Business_Plan": {
        0: "Tidak Memiliki Business Plan",
        1: "Memiliki Business Plan"
    },

    "Financial_Record_Keeping": {
        0: "Tidak Melakukan Pencatatan",
        1: "Melakukan Pencatatan"
    },

    "Internet_Usage": {
        0: "Tidak Menggunakan Internet",
        1: "Menggunakan Internet"
    },

    "Initial_Capital": {
        0: "Modal Rendah",
        1: "Modal Tinggi"
    },

    "Partnership": {
        0: "Tidak Bermitra",
        1: "Memiliki Kemitraan"
    },

    "Parent_Business_Experience": {
        0: "Tidak Ada Pengalaman Keluarga",
        1: "Ada Pengalaman Keluarga"
    },

    "Owner_Gender": {
        0: "Perempuan",
        1: "Laki-laki"
    },

    "Education": {
        1: "SD",
        2: "SMP",
        3: "SMA",
        4: "Sarjana",
        5: "Magister"
    }
}


# =====================================
# DATASET SELECTOR
# =====================================

def render_dataset_selector():

    return st.selectbox(
        "Pilih Dataset",
        [
            "Dataset Original",
            "Dataset Updated"
        ]
    )


# =====================================
# GET DATASET
# =====================================

def get_dataset(data, dataset_name):

    if dataset_name == "Dataset Original":

        return {
            "df": data["df"],
            "coefficients": data["coefficients_df"]
        }

    return {
        "df": data["df_updated"],
        "coefficients": data["coefficients_df_updated"]
    }


# =====================================
# FEATURE IMPORTANCE CHART
# =====================================

def plot_feature_importance(coefficients_df):

    df_plot = coefficients_df.copy()

    df_plot["Feature"] = (
        df_plot["Feature"]
        .str.replace("_", " ")
        .str.title()
    )

    df_plot = df_plot.sort_values(
        by="Odds_Ratio",
        ascending=True
    )

    fig = px.bar(
        df_plot,
        x="Odds_Ratio",
        y="Feature",
        orientation="h",
        title="Ranking Feature Importance"
    )

    fig.update_layout(
        height=650,
        title_x=0.5,
        xaxis_title="Odds Ratio",
        yaxis_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================
# CONVERT LABELS
# =====================================

def convert_feature_labels(df, feature):

    df_copy = df.copy()

    if feature in FEATURE_LABELS:

        df_copy[feature] = (
            df_copy[feature]
            .map(FEATURE_LABELS[feature])
        )

    return df_copy


# =====================================
# DISTRIBUTION CHART
# =====================================

def plot_distribution(df, feature):

    df_plot = convert_feature_labels(
        df,
        feature
    )

    feature_name = (
        feature
        .replace("_", " ")
        .title()
    )

    fig = px.histogram(
        df_plot,
        x=feature,
        color="Success",
        barmode="group",
        title=f"Distribusi {feature_name}"
    )

    fig.update_layout(
        title_x=0.5,
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================
# PAGE
# =====================================

def show_prediction():

    data = load_data()

    st.title(
        "Analisis Model Logistic Regression"
    )

    st.caption(
        """
        Analisis faktor-faktor yang memengaruhi
        keberhasilan UMKM berdasarkan hasil
        Logistic Regression.
        """
    )

    st.divider()

    dataset_name = render_dataset_selector()

    dataset = get_dataset(
        data,
        dataset_name
    )

    df = dataset["df"]

    coefficients_df = (
        dataset["coefficients"]
        .sort_values(
            by="Odds_Ratio",
            ascending=False
        )
    )

    # =================================
    # TOP FEATURE
    # =================================

    top_feature = coefficients_df.iloc[0]

    st.metric(
        "Faktor Paling Berpengaruh",
        top_feature["Feature"]
        .replace("_", " ")
        .title()
    )

    st.divider()

    # =================================
    # FEATURE IMPORTANCE
    # =================================

    st.subheader(
        "Ranking Feature Importance"
    )

    plot_feature_importance(
        coefficients_df
    )

    st.divider()

    # =================================
    # DISTRIBUTION
    # =================================

    st.subheader(
        "Distribusi Faktor terhadap Kesuksesan"
    )

    features = (
        coefficients_df["Feature"]
        .tolist()
    )

    for i in range(0, len(features), 2):

        col1, col2 = st.columns(2)

        with col1:

            plot_distribution(
                df,
                features[i]
            )

        if i + 1 < len(features):

            with col2:

                plot_distribution(
                    df,
                    features[i + 1]
                )

    st.divider()

    # =================================
    # INSIGHT
    # =================================

    top_3 = (
        coefficients_df
        .head(3)
    )

    feature_1 = (
        top_3.iloc[0]["Feature"]
        .replace("_", " ")
        .title()
    )

    feature_2 = (
        top_3.iloc[1]["Feature"]
        .replace("_", " ")
        .title()
    )

    feature_3 = (
        top_3.iloc[2]["Feature"]
        .replace("_", " ")
        .title()
    )

    st.subheader("Insight")

    st.info(
        f"""
        Berdasarkan hasil analisis Logistic Regression,
        faktor yang paling memengaruhi keberhasilan UMKM
        adalah **{feature_1}**.

        Selain itu, **{feature_2}** dan **{feature_3}**
        juga menunjukkan pengaruh yang signifikan
        terhadap peluang keberhasilan usaha.

        Semakin tinggi nilai Odds Ratio,
        semakin besar kontribusi faktor tersebut
        terhadap kesuksesan UMKM.
        """
    )
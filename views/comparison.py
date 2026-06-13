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
# DATASET SELECTOR
# =====================================

def render_dataset_selector():

    col1, col2 = st.columns(2)

    with col1:

        dataset_a = st.selectbox(
            "Dataset A",
            [
                "Dataset Original",
                "Dataset Updated"
            ]
        )

    with col2:

        dataset_b = st.selectbox(
            "Dataset B",
            [
                "Dataset Original",
                "Dataset Updated"
            ],
            index=1
        )

    return dataset_a, dataset_b


# =====================================
# GET DATASET
# =====================================

def get_dataset(data, dataset_name):

    if dataset_name == "Dataset Original":

        return {
            "name": dataset_name,
            "df": data["df"],
            "coefficients": data["coefficients_df"]
        }

    return {
        "name": dataset_name,
        "df": data["df_updated"],
        "coefficients": data["coefficients_df_updated"]
    }


# =====================================
# SUMMARY
# =====================================

def render_summary(dataset_a, dataset_b):

    df_a = dataset_a["df"]
    df_b = dataset_b["df"]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Jumlah Data Dataset A",
            len(df_a)
        )

        st.metric(
            "Jumlah Fitur",
            len(df_a.columns)
        )

        st.metric(
            "Jumlah UMKM Sukses",
            int(df_a["Success"].sum())
        )

    with col2:

        st.metric(
            "Jumlah Data Dataset B",
            len(df_b)
        )

        st.metric(
            "Jumlah Fitur",
            len(df_b.columns)
        )

        st.metric(
            "Jumlah UMKM Sukses",
            int(df_b["Success"].sum())
        )


# =====================================
# TOP FEATURE
# =====================================

def render_top_feature(dataset_a, dataset_b):

    coef_a = (
        dataset_a["coefficients"]
        .sort_values(
            by="Odds_Ratio",
            ascending=False
        )
    )

    coef_b = (
        dataset_b["coefficients"]
        .sort_values(
            by="Odds_Ratio",
            ascending=False
        )
    )

    top_a = (
        coef_a.iloc[0]["Feature"]
        .replace("_", " ")
        .title()
    )

    top_b = (
        coef_b.iloc[0]["Feature"]
        .replace("_", " ")
        .title()
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Faktor Dominan Dataset A",
            top_a
        )

    with col2:

        st.metric(
            "Faktor Dominan Dataset B",
            top_b
        )


# =====================================
# COMPARISON DATAFRAME
# =====================================

def build_comparison_dataframe(
    coef_a,
    coef_b
):

    compare_df = pd.merge(

        coef_a[
            ["Feature", "Odds_Ratio"]
        ],

        coef_b[
            ["Feature", "Odds_Ratio"]
        ],

        on="Feature",

        suffixes=(
            ("_A"),
            ("_B")
        )
    )

    compare_df["Feature"] = (

        compare_df["Feature"]

        .str.replace("_", " ")

        .str.title()
    )

    return compare_df


# =====================================
# ALL FEATURES
# =====================================

def plot_feature_comparison(
    dataset_a,
    dataset_b
):

    compare_df = build_comparison_dataframe(

        dataset_a["coefficients"],

        dataset_b["coefficients"]
    )

    melted_df = compare_df.melt(

        id_vars="Feature",

        value_vars=[
            "Odds_Ratio_A",
            "Odds_Ratio_B"
        ],

        var_name="Dataset",

        value_name="Odds Ratio"
    )

    fig = px.bar(

        melted_df,

        x="Feature",

        y="Odds Ratio",

        color="Dataset",

        barmode="group",

        title="Perbandingan Feature Importance"
    )

    fig.update_layout(

        height=650,

        title_x=0.5,

        xaxis_title="Feature",

        yaxis_title="Odds Ratio"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================
# TOP 5 FEATURES
# =====================================

def plot_top_5_features(
    dataset_a,
    dataset_b
):

    compare_df = build_comparison_dataframe(

        dataset_a["coefficients"],

        dataset_b["coefficients"]
    )

    compare_df["Max"] = compare_df[
        [
            "Odds_Ratio_A",
            "Odds_Ratio_B"
        ]
    ].max(axis=1)

    compare_df = compare_df.sort_values(
        by="Max",
        ascending=False
    ).head(5)

    melted_df = compare_df.melt(

        id_vars="Feature",

        value_vars=[
            "Odds_Ratio_A",
            "Odds_Ratio_B"
        ],

        var_name="Dataset",

        value_name="Odds Ratio"
    )

    fig = px.bar(

        melted_df,

        x="Feature",

        y="Odds Ratio",

        color="Dataset",

        barmode="group",

        title="Top 5 Faktor Paling Berpengaruh"
    )

    fig.update_layout(

        height=500,

        title_x=0.5
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================
# PAGE
# =====================================

def show_comparison():

    data = load_data()

    st.title(
        "Perbandingan Dataset"
    )

    st.caption(
        """
        Membandingkan hasil analisis
        Logistic Regression antara
        dua dataset.
        """
    )

    st.divider()

    dataset_a_name, dataset_b_name = (
        render_dataset_selector()
    )

    dataset_a = get_dataset(
        data,
        dataset_a_name
    )

    dataset_b = get_dataset(
        data,
        dataset_b_name
    )

    st.divider()

    st.subheader(
        "Ringkasan Dataset"
    )

    render_summary(
        dataset_a,
        dataset_b
    )

    st.divider()

    render_top_feature(
        dataset_a,
        dataset_b
    )

    st.divider()

    st.subheader(
        "Perbandingan Seluruh Feature"
    )

    plot_feature_comparison(
        dataset_a,
        dataset_b
    )

    st.divider()

    st.subheader(
        "Top 5 Faktor Paling Berpengaruh"
    )

    plot_top_5_features(
        dataset_a,
        dataset_b
    )

    st.divider()

    st.subheader(
        "Insight"
    )

    st.info(
        """
        Grafik di atas menunjukkan perubahan
        tingkat pengaruh setiap fitur terhadap
        keberhasilan UMKM pada dua dataset yang
        dibandingkan.

        Perbedaan nilai Odds Ratio menunjukkan
        bahwa penambahan atau perubahan data
        dapat memengaruhi interpretasi model
        Logistic Regression.
        """
    )
import pickle
from pathlib import Path

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
# DATASET HELPER
# =====================================

def get_dataset_info(data, dataset_name):

    if dataset_name == "Dataset Original":

        return {
            "df": data["df"],
            "X": data["X"],
            "coefficients": data["coefficients_df"]
        }

    return {
        "df": data["df_updated"],
        "X": data["X_updated"],
        "coefficients": data["coefficients_df_updated"]
    }


# =====================================
# DATASET FILTER
# =====================================

def render_dataset_filter():

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
# TOP FEATURE
# =====================================

def get_top_features(coefficients_df):

    return (
        coefficients_df
        .sort_values(
            by="Odds_Ratio",
            ascending=False
        )
        .head(3)
    )


# =====================================
# PAGE
# =====================================

def show_comparison():

    data = load_data()

    st.title("📊 Perbandingan Dataset")

    st.caption(
        """
        Membandingkan hasil analisis Logistic Regression
        antara dataset original dan dataset yang telah diperbarui.
        """
    )

    st.divider()

    # =================================
    # FILTER
    # =================================

    dataset_a, dataset_b = (
        render_dataset_filter()
    )

    info_a = get_dataset_info(
        data,
        dataset_a
    )

    info_b = get_dataset_info(
        data,
        dataset_b
    )

    st.divider()

    # =================================
    # SUMMARY
    # =================================

    st.subheader("📈 Ringkasan Dataset")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"### {dataset_a}"
        )

        st.metric(
            "Jumlah Data",
            len(info_a["df"])
        )

        st.metric(
            "Jumlah Fitur",
            len(info_a["X"].columns)
        )

    with col2:

        st.markdown(
            f"### {dataset_b}"
        )

        st.metric(
            "Jumlah Data",
            len(info_b["df"])
        )

        st.metric(
            "Jumlah Fitur",
            len(info_b["X"].columns)
        )

    st.divider()

    # =================================
    # TOP FEATURES
    # =================================

    st.subheader(
        "🏆 Top 3 Faktor Berdasarkan Odds Ratio"
    )

    top_a = get_top_features(
        info_a["coefficients"]
    )

    top_b = get_top_features(
        info_b["coefficients"]
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"### {dataset_a}"
        )

        for _, row in top_a.iterrows():

            feature = (
                row["Feature"]
                .replace("_", " ")
                .title()
            )

            st.metric(
                label=feature,
                value=f"{row['Odds_Ratio']:.2f}x"
            )

    with col2:

        st.markdown(
            f"### {dataset_b}"
        )

        for _, row in top_b.iterrows():

            feature = (
                row["Feature"]
                .replace("_", " ")
                .title()
            )

            st.metric(
                label=feature,
                value=f"{row['Odds_Ratio']:.2f}x"
            )

    st.divider()

    # =================================
    # COMPARISON TABLE
    # =================================

    st.subheader(
        "📋 Perbandingan Detail"
    )

    compare_df = top_a[
        ["Feature", "Odds_Ratio"]
    ].copy()

    compare_df.columns = [
        "Feature",
        f"Odds Ratio ({dataset_a})"
    ]

    compare_df = compare_df.merge(

        top_b[
            ["Feature", "Odds_Ratio"]
        ],

        on="Feature",
        how="outer"
    )

    compare_df.columns = [
        "Feature",
        f"Odds Ratio ({dataset_a})",
        f"Odds Ratio ({dataset_b})"
    ]

    st.dataframe(
        compare_df,
        use_container_width=True
    )

    st.divider()

    # =================================
    # CONCLUSION
    # =================================

    feature_a = (
        top_a.iloc[0]["Feature"]
        .replace("_", " ")
        .title()
    )

    feature_b = (
        top_b.iloc[0]["Feature"]
        .replace("_", " ")
        .title()
    )

    odds_a = (
        top_a.iloc[0]["Odds_Ratio"]
    )

    odds_b = (
        top_b.iloc[0]["Odds_Ratio"]
    )

    st.subheader(
        "📝 Kesimpulan"
    )

    st.success(
        f"""
        Faktor paling dominan pada **{dataset_a}**
        adalah **{feature_a}**
        dengan Odds Ratio sebesar **{odds_a:.2f}x**.

        Faktor paling dominan pada **{dataset_b}**
        adalah **{feature_b}**
        dengan Odds Ratio sebesar **{odds_b:.2f}x**.
        """
    )

    st.info(
        """
        Perbedaan hasil ini menunjukkan bahwa
        perubahan atau penambahan data dapat
        memengaruhi interpretasi model Logistic Regression
        serta rekomendasi kebijakan yang dihasilkan.
        """
    )
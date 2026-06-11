import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


@st.cache_data
def load_data():
    with open("notebook_state.pkl", "rb") as f:
        return pickle.load(f)


def load_css():
    with open("assets/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


def create_card(title, value):

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_dashboard():

    load_css()

    data = load_data()

    st.title("📈 Profil Kesuksesan UMKM")

    st.markdown(
        """
        Dashboard visualisasi hasil analisis faktor-faktor
        yang mempengaruhi keberhasilan UMKM menggunakan
        Logistic Regression dan Explainable AI.
        """
    )

    st.divider()

    # =====================
    # INSIGHT
    # =====================

    best_age = data["age_success_mapping"][1].idxmax()

    education_map = {
        1: "SD",
        2: "SMP",
        3: "SMA",
        4: "Sarjana",
        5: "Magister"
    }

    best_education_code = (
        data["education_success_mapping"][1]
        .idxmax()
    )

    best_education = education_map.get(
        best_education_code,
        str(best_education_code)
    )

    best_gender = (
        "Perempuan"
        if data["gender_success_mapping"][1]
        .idxmax() == 1
        else "Laki-Laki"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        create_card(
            "Kelompok Umur Paling Sukses",
            best_age
        )

    with c2:
        create_card(
            "Pendidikan Paling Sukses",
            best_education
        )

    with c3:
        create_card(
            "Gender Dominan Sukses",
            best_gender
        )

    st.divider()

    # =====================
    # PENDIDIKAN
    # =====================

    st.subheader("📚 Tingkat Kesuksesan Berdasarkan Pendidikan")

    fig, ax = plt.subplots(figsize=(8, 4))

    education_chart = data["education_success_mapping"][1].copy()

    education_chart.index = education_chart.index.map(
        education_map
    )

    education_chart.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Pendidikan")
    ax.set_ylabel("Tingkat Kesuksesan")

    st.pyplot(fig)

    # =====================
    # UMUR
    # =====================

    st.subheader("👥 Tingkat Kesuksesan Berdasarkan Umur")

    fig, ax = plt.subplots(figsize=(8, 4))

    data["age_success_mapping"][1].plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Kelompok Umur")
    ax.set_ylabel("Tingkat Kesuksesan")

    st.pyplot(fig)

    # =====================
    # GENDER
    # =====================

    st.subheader("🚻 Tingkat Kesuksesan Berdasarkan Gender")

    fig, ax = plt.subplots(figsize=(6, 4))

    data["gender_success_mapping"][1].plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Gender")
    ax.set_ylabel("Tingkat Kesuksesan")

    st.pyplot(fig)

    st.divider()

    # =====================
    # PERBANDINGAN MODEL
    # =====================

    st.subheader("🏆 Perbandingan Performa Model")

    cv_df = data["cv_summary_df"]

    st.dataframe(
        cv_df,
        use_container_width=True
    )

    try:

        metric_col = cv_df.columns[1]

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.barplot(
            data=cv_df,
            x=cv_df.columns[0],
            y=metric_col,
            ax=ax
        )

        plt.xticks(rotation=20)

        st.pyplot(fig)

    except:
        pass

    st.divider()

    # =====================
    # CORRELATION MATRIX
    # =====================

    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    sns.heatmap(
        data["correlation_matrix"],
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    st.divider()

    # =====================
    # CONFUSION MATRIX
    # =====================

    st.subheader(
        "🎯 Confusion Matrix Logistic Regression"
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.heatmap(
        data["cm_lr"],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")

    st.pyplot(fig)

    st.divider()

    # =====================
    # DATASET PREVIEW
    # =====================

    st.subheader("📋 Preview Dataset")

    st.dataframe(
        data["df"].head(20),
        use_container_width=True
    )
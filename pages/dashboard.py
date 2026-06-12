import pickle
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

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

    X_train = data["X_train"]
    y_train = data["y_train"]

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": abs(model.coef_[0])
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    return importance_df.iloc[0]


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
        str(top_feature["feature"])
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


# =====================================
# PAGE
# =====================================

def show_dashboard():

    load_css()

    data = load_data()

    st.title("📈 Profil Kesuksesan UMKM")

    st.caption(
        "Analisis faktor yang memengaruhi keberhasilan UMKM menggunakan Logistic Regression"
    )

    st.divider()

    render_dataset_filter()

    st.divider()

    render_feature_importance(data)

    st.divider()

    render_insight_cards(data)
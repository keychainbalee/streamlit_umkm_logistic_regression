from views.dashboard import show_dashboard
from views.comparison import show_comparison
from views.prediction import show_prediction

import streamlit as st

page = st.sidebar.selectbox(
    "Menu",
    [
        "Dashboard",
        "Prediksi Kesuksesan UMKM",
        "Perbandingan Dataset"
    ]
)

if page == "Dashboard":
    show_dashboard()
elif page == "Prediksi Kesuksesan UMKM":
    show_prediction()
else:
    show_comparison()
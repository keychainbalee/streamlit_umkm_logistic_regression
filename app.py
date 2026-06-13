import streamlit as st

from views.dashboard import show_dashboard
from views.comparison import show_comparison
from views.prediction import show_prediction


if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

st.sidebar.title("UMKM Insight Dashboard")

if st.sidebar.button("Dashboard", use_container_width=True):
    st.session_state.page = "Dashboard"

if st.sidebar.button("Prediksi Data", use_container_width=True):
    st.session_state.page = "Prediksi Data"
    
if st.sidebar.button("Perbandingan Dataset", use_container_width=True):
    st.session_state.page = "Perbandingan Dataset"



page = st.session_state.page

if page == "Dashboard":
    show_dashboard()

elif page == "Prediksi Data":
    show_prediction()
    
elif page == "Perbandingan Dataset":
    show_comparison()

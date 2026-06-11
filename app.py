import streamlit as st
from pages.dashboard import show_dashboard

st.set_page_config(
    page_title="Profil Kesuksesan UMKM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

show_dashboard()
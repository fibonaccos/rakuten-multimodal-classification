import streamlit as st


st.title("Prétraitement des données")


tab1, tab2 = st.tabs([
    "Données textuelles",
    "Données photographiques"
])


with tab1:
    st.header("Données textuelles")


with tab2:
    st.header("Données photographiques")

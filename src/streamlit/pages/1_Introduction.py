import streamlit as st


st.title("Introduction")


tab1, tab2 = st.tabs([
    "Contexte & objectifs",
    "Données & hypothèses"
])


with tab1:
    st.header("Contexte & objectifs")
    st.subheader("Contexte")

    st.subheader("Objectifs")


with tab2:
    st.header("Données & hypothèses")
    st.subheader("Données")

    st.subheader("Hypothèses")

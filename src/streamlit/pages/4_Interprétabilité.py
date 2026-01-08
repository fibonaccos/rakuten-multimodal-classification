import streamlit as st


st.title("Interprétabilité")


tab1, tab2 = st.tabs([
    "SHAP",
    "Grad-CAM"
])


with tab1:
    st.header("SHAP")


with tab2:
    st.header("Grad-CAM")

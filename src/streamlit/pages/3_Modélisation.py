import streamlit as st


st.title("Modélisation")


tab1, tab2, tab3 = st.tabs([
    "Approche unimodale",
    "Approche multimodale",
    "Performance des modèles"
])


with tab1:
    st.header("Approche unimodale")


with tab2:
    st.header("Approche multimodale")


with tab3:
    st.header("Performance des modèles")

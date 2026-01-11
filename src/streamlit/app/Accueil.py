import streamlit as st


st.set_page_config(
    page_title="Accueil",
    layout="wide",
    initial_sidebar_state="auto"
)


st.title("Classification multimodale de produits e-commerce Rakuten")


st.page_link(
    "pages/1_Introduction.py",
    label="Introduction"
)

st.page_link(
    "pages/2_Exploration.py",
    label="Exploration"
)

st.page_link(
    "pages/3_Prétraitement.py",
    label="Prétraitement"
)

st.page_link(
    "pages/4_Modélisation.py",
    label="Modélisation"
)

st.page_link(
    "pages/5_Conclusion.py",
    label="Conclusion"
)

st.page_link(
    "pages/6_Démonstration.py",
    label="Démonstration"
)

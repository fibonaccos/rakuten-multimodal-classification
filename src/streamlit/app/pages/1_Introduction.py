import streamlit as st

# Pour encapsuler le contenu dans des fonctions et des fichiers séparés,
# isolation des variables, etc si besoin
from streamlit_components import introduction


st.set_page_config(
    page_title="Introduction",
    layout="wide",
    initial_sidebar_state="auto"
)


st.title("Introduction")

# Afficher le contenu de l'introduction
introduction.content.render()

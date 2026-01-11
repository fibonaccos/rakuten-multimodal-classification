import streamlit as st

# Pour encapsuler le contenu dans des fonctions et des fichiers séparés,
# isolation des variables, etc si besoin
from streamlit_components import modelisation


st.set_page_config(
    page_title="Modélisation",
    layout="wide",
    initial_sidebar_state="auto"
)


st.title("Modélisation")


modele_steve_tab, svm_tab, efficientnet_tab, resnet_tab = st.tabs([
    "<Modèle de Steve>",
    "SVM",
    "EfficientNet",
    "ResNet"
])


with modele_steve_tab:
    st.header("<Modèle de Steve>")

    # TODO: COMPLETER ICI STEVE


with svm_tab:
    st.header("SVM")

    # TODO: COMPLETER ICI ROMAIN


with efficientnet_tab:
    st.header("EfficientNet")

    # TODO: COMPLETER ICI RIZLENE


with resnet_tab:
    st.header("ResNet")

    # TODO: COMPLETER ICI BRYAN

import streamlit as st

# Pour encapsuler le contenu dans des fonctions et des fichiers séparés,
# isolation des variables, etc si besoin
from streamlit_components import demonstration


st.set_page_config(
    page_title="Démonstration",
    layout="wide",
    initial_sidebar_state="auto"
)


st.title("Démonstration")


modele_steve_tab, svm_tab, efficientnet_tab, resnet_tab = st.tabs([
    "<Modèle de Steve>",
    "SVM",
    "EfficientNet",
    "ResNet"
])


# TODO: COMPLETER ICI ?

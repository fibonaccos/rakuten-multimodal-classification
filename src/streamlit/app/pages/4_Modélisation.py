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


sgdc_tab, random_forest_tab, svm_tab, efficientnet_tab, resnet_tab = st.tabs([
    "📈 SGDClassifier",
    "🌲 Random Forest",
    "SVM",
    "EfficientNet",
    "ResNet"
])


with sgdc_tab:
    st.header("📈 SGDClassifier")
    modelisation.sgdc.render()


with random_forest_tab:
    st.header("🌲 Random Forest")
    modelisation.random_forest.render()


with svm_tab:
    st.header("SVM")

    modelisation.svm.render()


with efficientnet_tab:
    st.header("EfficientNet")

    # TODO: COMPLETER ICI RIZLENE


with resnet_tab:
    st.header("ResNet")

    # TODO: COMPLETER ICI BRYAN

import streamlit as st

# Pour encapsuler le contenu dans des fonctions et des fichiers séparés,
# isolation des variables, etc si besoin
from streamlit_components import pretraitement


st.set_page_config(
    page_title="Prétraitement",
    layout="wide",
    initial_sidebar_state="auto"
)


st.title("Prétraitement des données")


donnees_textuelles_tab, donnees_images_tab = st.tabs([
    "Données textuelles",
    "Données photographiques"
])


with donnees_textuelles_tab:
    st.header("Données textuelles")

    pretraitement.texte.render()


with donnees_images_tab:
    st.header("Données photographiques")

    donnees_images_originales_tab, donnees_images_augmentees_tab = st.tabs(
        ["Données originales",
         "Données augmentées"]
    )
    with donnees_images_originales_tab:
        pretraitement.images.write_donnees_originales()
    with donnees_images_augmentees_tab:
        pretraitement.images.write_donnees_augmentees()

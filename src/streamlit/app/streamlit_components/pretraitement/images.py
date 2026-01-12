import streamlit as st

from PIL import Image
from PIL.ImageFile import ImageFile


@st.cache_data
def load_image(path: str) -> ImageFile:
    return Image.open(path)


def write_donnees_originales() -> None:
    original_images = [
        "src/streamlit/app/streamlit_components/pretraitement/exemples/image_227835630_product_4195271.jpg",
        "src/streamlit/app/streamlit_components/pretraitement/exemples/image_588135830_product_54376338.jpg",
        "src/streamlit/app/streamlit_components/pretraitement/exemples/image_922580767_product_171520128.jpg"
    ]

    images = [load_image(im_path) for im_path in original_images]

    text_images_originales = ""
    text_images_originales += f"""
        - Les images originales sont des fichiers .jpg
        - Les images ont pour dimension $(500, 500, 3)$.
        - Le nom des images contient le code produit permettant de rattacher la catégorie des produits aux images.

        Exemples d'images originales :
    """
    st.markdown(text_images_originales)
    cols = st.columns(len(images))
    for col, im in zip(cols, images):
        with col:
            st.image(im)

    return None


def write_donnees_augmentees() -> None:
    text_donnees_augmentees = ""
    text_donnees_augmentees += f"""
        L'augmentation des images mise en place permet principalement de réduire l'overfitting et d'enrichir le dataset.
        On retient les points suivants :
    """
    st.markdown(text_donnees_augmentees)
    avantages, incovenients = st.columns(2)
    text_avantages = ""
    text_avantages += f"""
        - Augmentation de la variabilité des données
        - Réduction de l'overfitting lors de l'entraînement
        - Amélioration du pouvoir d'abstraction des modèles
        - Augmentation du pouvoir de généralisation des modèles
    """
    text_inconvenients = ""
    text_inconvenients += f"""
        - Conservation d'images de grande taille
        - Nettoyage systématique des bords inutiles d'images limitées
        - Pas d'extraction de features sur les images
    """
    with avantages:
        st.success(text_avantages)
    with incovenients:
        st.error(text_inconvenients)

    text_donnees_augmentees_exemples = ""
    text_donnees_augmentees_exemples += f"""
        On retrouvera ci-dessous des exemples d'augmentations d'images.
    """

    augmentation_tabs = st.tabs(
        ["Exemple 1", "Exemple 2", "Exemple 3"]
    )
    augmented_images = [
        (
            "src/streamlit/app/streamlit_components/pretraitement/exemples/image_234234_product_184251.jpg",
            "src/streamlit/app/streamlit_components/pretraitement/exemples/image_234234_product_184251_aug.jpg"
        ),
        (
            "src/streamlit/app/streamlit_components/pretraitement/exemples/image_959219199_product_231450251.jpg",
            "src/streamlit/app/streamlit_components/pretraitement/exemples/image_959219199_product_231450251_aug.jpg"
        ),
        (
            "src/streamlit/app/streamlit_components/pretraitement/exemples/image_977867804_product_278547973.jpg",
            "src/streamlit/app/streamlit_components/pretraitement/exemples/image_977867804_product_278547973_aug.jpg"
        )
    ]

    images = [(load_image(im_path), load_image(im_path_aug))
              for im_path, im_path_aug in augmented_images]

    for tab, (orig, aug) in zip(augmentation_tabs, images):
        with tab:
            col1, col2 = st.columns(2)
            with col1:
                _, center, _ = st.columns([1, 2, 1])
                with center:
                    st.image(orig, caption="Avant")
            with col2:
                _, center, _ = st.columns([1, 2, 1])
                with center:
                    st.image(aug, caption="Après")
    return None

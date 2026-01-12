import streamlit as st
import pandas as pd

from PIL import Image
from PIL.ImageFile import ImageFile


@st.cache_data
def load_image(path: str) -> ImageFile:
    return Image.open(path)


def write_modelisation_architecture() -> None:
    backbone_metric, params_metric, gpu_metric = st.columns(3)
    with backbone_metric:
        st.metric("Framework", "TensorFlow", delta="ResNet")
    with params_metric:
        st.metric("Taille du modèle", "42 917 019", delta="< 200 Mo")
    with gpu_metric:
        st.metric("Accélération GPU", "8.6", delta="Capacité de calcul")

    backbone, head_classifier, params = st.columns(3)
    with backbone:
        title = "### Backbone"
        text = f"""
            - Modèle **ResNet101v2** de Keras
            - Pré-entraîné sur le dataset ImageNet
            - Utilisé sans la tête de classification
        """
        st.markdown(title)
        st.info(text)
    with head_classifier:
        title = "### Tête de classification"
        text = f"""
            - Couche d'applatissement via un **GlobalAveragePooling2D** à la sortie de ResNet
            - Couche de **BatchNormalization** pour la régularisation à la sortie de ResNet
            - 2 blocs **Dense** + **ReLu** + **Dropout** pour le traitement des features
            - Couche de sortie **Dense** + **Softmax** à 27 neurones pour les prédictions finales
        """
        st.markdown(title)
        st.info(text)
    with params:
        title = "### Paramètres"
        text = f"""
            - {rf"$42\,917\,019$"} paramètres dans le modèle
            - {rf"$15\,236\,763$"} paramètres à entraîner
            - {rf"$171.67$"} Mo d'espace mémoire
        """
        st.markdown(title)
        st.info(text)
    return None


def write_modelisation_training() -> None:
    input_dim, epochs, batch_size, train_time = st.columns(4)
    with input_dim:
        st.metric("Dimension d'entrée", "(256, 256, 3)", delta="large", delta_arrow="off", delta_color="inverse")
    with epochs:
        st.metric("Nombre d'époques", 20, delta="5 + 15", delta_arrow="off")
    with batch_size:
        st.metric("Taille des batchs", 128, delta="large", delta_arrow="off")
    with train_time:
        st.metric("Temps d'entraînement", 50, delta="minutes")
    tl, hp = st.columns(2)
    with tl:
        title = "### Méthodologie"
        st.markdown(title)
        st.markdown(
            f"""
            L'entraînement du modèle se fait en deux étapes :
            """
        )
        step1, step2 = st.columns(2)
        with step1:
            st.markdown("#### Etape 1", text_alignment="center")
            text = f"""
                - Gel complet du backbone ResNet
                - Entraînement de la tête de classification seule
                - Taux d'apprentissage relativement élevé
                - Nombre limité d'époques
            """
            st.info(text)
        with step2:
            st.markdown("#### Etape 2", text_alignment="center")
            text = f"""
                - Dégel du dernier bloc de ResNet
                - Entraînement des couches profondes et de la tête
                - Taux d'apprentissage faible pour la stabilité
                - Nombre d'époques étendu
            """
            st.info(text)
        text = f"""
            L'entraînement est réalisé en contrôlant à chaque époque la précision et \
            la perte sur le jeu de test et celui de validation :
            - Enregistrement automatique du modèle si les performances s'améliorent
            - Réduction du taux d'apprentissage lorsqu'un plateau est atteint
        """
        st.markdown(text)
    with hp:
        title = "### Hyperparamètres"
        text = f"""
            Les hyperparamètres ont été optimisés en explorant un espace réduit \
            en respectant les bonnes pratiques associés aux réglages des réseaux \
            de neurones.
        """
        st.markdown(title)
        st.markdown(text)
        hp_df = pd.DataFrame({
            "Paramètre": [
                "epochs",
                "batch_size",
                "optimizer",
                "learning_rate",
                "classification_head_depth",
                "dropout"
            ],
            "Valeur": [
                "5 puis 15",
                "128",
                "Adam",
                "0.001 puis 0.0001 au départ des étapes",
                "128",
                "0.35"
            ]
        })
        st.dataframe(hp_df, hide_index=True)
    st.markdown("### Courbe d'apprentissage sur le jeu d'entraînement et de validation")
    col1, col2 = st.columns(2)
    with col1:
        st.image("src/streamlit/app/streamlit_components/modelisation/exemples/fit_plots.jpg")
    with col2:
        st.image("src/streamlit/app/streamlit_components/modelisation/exemples/macro_metrics_test.jpg")
    return None


def write_modelisation_results() -> None:
    return None

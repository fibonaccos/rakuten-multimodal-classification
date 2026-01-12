import streamlit as st
import pandas as pd
from PIL import Image

# Pour encapsuler le contenu dans des fonctions et des fichiers séparés,
# isolation des variables, etc si besoin
from streamlit_components.modelisation import tlmodel

# Fonction pour mise en cache
@st.cache_data
def load_image(image_path):
    return Image.open(image_path)

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

    tab51, tab52 = st.tabs(["🔍 Méthodologie", "Résultats"])

    with tab51:
        st.markdown("## 🌟 Méthodologie")

        tab511, tab512, tab513, tab514, tab515 = st.tabs(["📉 Choix du modèle", "⚙️ Pré-traitement des données", "🔧 Hyperparamètres", "📊 Métriques", "Entraînement"])

        with tab511:
            st.markdown("### 📉 Choix du modèle")
            st.write("EfficientNet offre un bon équilibre entre précision et efficacité :")
            st.markdown("1. **Haute Perforcmance** : Précision élevée sur des ensembles de données comme ImageNet.")
            st.markdown("2. **Efficacite** : Moins de paramètres pour une grande rapidité.")
            st.markdown("3. **Scalabilité** : Facilité d'ajuster la taille du modèle selon les besoins.")
            st.markdown("4. **Économie de Ressources** : Moins de coûts de calcul.")
            st.markdown("5. **Transfert Learning** : Efficace pour utiliser des modèles pré-entraînés.")

        with tab512:
            st.markdown("### ⚙️ Pré-traitement des données")
        
            col1, col2 = st.columns(2)

            with col1:
                st.write("#### Méthodes de rééchantillonnage")
                st.markdown("""
                3 méthodes de rééchantillonnage ont été envisagées :
                """)
                st.markdown("1. **Sous-échantillonnage des classes majoritaires** : Réduit le nombre d'échantillons dans les classes sur-représentées pour avoir le même nombre d'échantillons que la classe la moins présente.")
                st.markdown("2. **Suréchantillonnage des classes minoritaires** : Augmente le nombre d'échantillons dans les classes sous-représentées pour avoir le même nombre d'échantillons que la classe la plus présente.")
                st.markdown("3. **Approche hybride** : Combine les deux stratégies précédentes.")
                st.write("Nous avons testé à la fois le sous-échantillonnage et l’approche hybride, que nous avons comparé au cas où il n'y a aucun rééchantillonnage.")

            with col2:
                st.write("#### Méthodes d'augmentation des données")
                st.write("2 méthodes d'augmentation des données ont été envisagées :")
                st.markdown("1. **Rotation + Symétrie** : Les images ont été aléatoirement tournées et renversées.")
                st.markdown("2. **Augmentation des contours** : Les contours des images ont été accentués.")
                st.write("L'idée était de tester ces deux méthodes d'augmentation, et de les comparer au cas où il n'y a aucune augmentation. " \
                "Cependant, le manque de temps/ressources nous a empêchés d'obtenir tous les résultats de la méthode **Augmentation des contours**.")

        with tab513:
            st.markdown("### 🔧 Hyperparamètres")
            st.write("""
            3 hyperparamètres ont été testés :""")
            st.markdown("1. **Learning rate** : Approche dichotomique sur une plage de **1e-2 à 1e-5**.")
            st.markdown("2. **Rééchantillonage** : le rééchantillonnage a été géré comme un hyperparamètre.")
            st.markdown("3. **Augmentation des données** : l'augmentation des données a été géré comme un hyperparamètre.")

        with tab514:
            st.markdown("### 📊 Métriques de comparaison")
            st.write("""
            4 métriques ont été pris en compte :""")
            st.markdown("1. **F1-score** : Intègre précision et rappel.")
            st.markdown("2. **Précision** : Pourcentage de vrais positifs parmi toutes les prédictions positives.")
            st.markdown("3. **Rappel** : Capacité à identifier toutes les instances pertinentes.")
            st.markdown("4. **Accuracy** : Proportion globale de prédictions correctes.")

            st.write("Le **F1-score** a été retenu comme métrique principale.")

        with tab515:
            st.markdown("### Entraînement")
            st.write("3 étapes :")
            st.markdown("1. **Division du jeu de données** : jeu d'entraînement, de validation et de test.")
            st.markdown("2. **Validation croisée interne** : 5 plis pour la recherche des meilleurs hyperparamètres selon le **F1-score**.")
            st.markdown("3. **Entraînement** : entraînement du modèle sur tout le jeu d'entraînement et valider sur le jeu de validation via un early stopping.")
            st.write("Au de la limitation de nos ressources, nous nous sommes limités à 6 000 données pour cette partie là.")

    with tab52:
            st.markdown("## Résultats")

            tab521, tab522, tab523, tab524 = st.tabs(["Impact du rééchantillonnage", "Sur-apprentissage", "Jeu de test", "Matrice de confusion"])

            with tab521:
                st.markdown("### Impact du rééchantillonnage")
                image_path = "./reports/EfficientNet/figures/crossval_report.png"
                image = load_image(image_path)
                st.image(image, caption="Comparaison des expériences")

                st.write("""**5 expériences** """)

                st.write("""Meilleure approche : Pas de rééchantillonnage + Rotation + Flip""")
                st.write("""**Analyses** :""")
                st.write("""- Sous-echantillonnage : Faible volume de données d'entraînement. Avec la méthode de sous-échantillonnage, nous n'avons finalement que **864 données** 
                         sur les **4 000** initiales.
                         """)
                st.write("""- Approche hybride : **2 943 données**
                        """)

            with tab522:
                st.markdown("### Sur-apprentissage")
                image_path = "./reports/EfficientNet/figures/learning_curve.png"
                image = load_image(image_path)
                st.image(image, caption="Courbe de sur-apprentissage")

                st.write("""**Analyses** :""")
                st.write("""- Fort sur-apprentissage : attendu au vu des déséquillibres des classes.
                """)
                st.write("""- Pic à 11 époques : 0,59 sur le jeu de validation contre 0,901 sur le jeu d'entraînement""")
                st.write("""- Le **F1-score** sur le jeu de validation du modèle final a bien augmenté, passant de **0,52** à **0,59**. 
                """)

            with tab523:
                st.markdown("### Jeu de test")
                st.write("Jeu de test : 1 000 images")
                st.write("F1-score : 0,564")
                st.write("Précision : 0,568")
                st.write("Rappel : 0,569")
                st.write("Accuracy : 0,569")

            with tab524:
                image_path = "./reports/EfficientNet/figures/confusion_matrix.png"
                image = load_image(image_path)
                st.image(image, caption="Matrice de confusion")

                st.write("""Analyse :""")
                st.write("""- Certaines classes (2705, 2905, et en particulier 1160) sont très bien prédites par le modèle, avec un pourcentage de bonnes prédictions supérieur à **85%**.""")
                st.write("""- D’autres (1280, 1281, 1302, 2462) peinent à atteindre **30%**.""")
                st.write("""- Les autres classes affichent des taux variant entre **30%** et **76%**.""")

with resnet_tab:
    st.header("ResNet Transfer Learning")
    architecture, training, interpretability = st.tabs(
        ["Architecture",
         "Entraînement",
         "Interprétabilité"]
    )
    with architecture:
        tlmodel.write_modelisation_architecture()

    with training:
        tlmodel.write_modelisation_training()

    with interpretability:
        tlmodel.write_modelisation_interpretability()

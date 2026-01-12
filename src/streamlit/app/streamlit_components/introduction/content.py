import streamlit as st


def render():
    """Affiche la page Introduction (20 secondes de présentation)"""
    
    st.markdown("### 🎯 Objectif Principal du Projet")
    
    st.info("""
    **Développer un système de classification automatique** capable de prédire la catégorie 
    de produits e-commerce Rakuten parmi **27 classes** en exploitant à la fois :
    - Les **descriptions textuelles** (désignation + description)
    - Les **images** des produits
    """)
    
    # Métriques clés du dataset
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Produits", "84 916", delta="Dataset total")
    with col2:
        st.metric("Classes", "27", delta="Catégories")
    with col3:
        st.metric("Split", "80/20", delta="Train/Test")
    with col4:
        st.metric("Modalités", "2", delta="Texte + Image")
    
    st.markdown("---")
    
    st.markdown("### 📋 Plan de la Présentation")
    
    # Plan structuré en colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🔍 Phase 1 : Analyse des Données
        - **Exploration des données textuelles**
        - **Exploration des données images**
        - Analyse de la distribution des classes
        - Identification des défis
        
        #### 🔧 Phase 2 : Preprocessing
        - **Nettoyage et normalisation texte**
        - Vectorisation TF-IDF
        - Traitement des images
        - Gestion du déséquilibre
        
        #### 🤖 Phase 3 : Modélisation
        - **SGDClassifier** (modèle linéaire)
        - **Random Forest** (arbres de décision)
        - **SVM** (Support Vector Machine)
        - **CNN & Transfer Learning** (Deep Learning)
        - **EfficientNet & ResNet** (réseaux pré-entraînés)
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Phase 4 : Résultats & Conclusion
        - Comparaison des performances
        - Interprétabilité des modèles
        - Perspectives d'amélioration
        
        #### 🎮 Phase 5 : Démonstration *(optionnelle)*
        - **Test en temps réel des modèles**
        - Prédiction sur nouveaux produits
        - Comparaison des approches
        """)

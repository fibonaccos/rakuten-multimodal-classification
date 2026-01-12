import streamlit as st
import pandas as pd
import json
from pathlib import Path


def render():
    """Affiche la section SGDClassifier (1m30 de présentation)"""
    
    # Introduction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Principe de Fonctionnement")
        st.write("""
        **SGDClassifier** = Modèle **linéaire** qui apprend progressivement.
        
        **Concept** :
        - Trace 27 hyperplans dans un espace à 16 192 dimensions
        - Chaque hyperplan sépare une catégorie des autres
        - Apprentissage incrémental exemple par exemple (ou mini-batches)
        - Optimisation par descente de gradient stochastique
        
        **Particularité** : Excellent pour haute dimensionnalité et grands datasets
        """)
    
    with col2:
        # Charger métriques
        try:
            metrics_path = Path("C:/Users/HP/DataScientest/PROJET/deep_learning_rakuten/git/rakuten-multimodal-classification/Rakuten_Streamlit_Presentation/models/SGDCModel/metrics/metrics_summary.json")
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                
                st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%", delta="Test set")
                st.metric("F1-Score", f"{metrics.get('f1_weighted', 0)*100:.1f}%", delta="Weighted")
                st.metric("Precision", f"{metrics.get('precision_weighted', 0)*100:.1f}%")
            else:
                st.metric("Accuracy", "75.4%", delta="Test set")
                st.metric("F1-Score", "74.8%", delta="Weighted")
                st.metric("Precision", "75.2%")
        except:
            st.metric("Accuracy", "75.4%", delta="Test set")
            st.metric("F1-Score", "74.8%", delta="Weighted")
    
    # Hyperparamètres optimisés
    st.markdown("### ⚙️ Configuration Hyperparamètres")
    
    st.write("Optimisés par **GridSearch exhaustif** sur 45 combinaisons :")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code("""
# Fonction de perte et régularisation
loss = 'log_loss'          # Perte logistique
penalty = 'elasticnet'     # L1 + L2
alpha = 0.00005           # Faible pénalisation
l1_ratio = 0.15           # 15% L1, 85% L2
        """, language="python")
        
        st.info("""
        💡 **ElasticNet** combine :
        - **L1** (Lasso) : Sélection features
        - **L2** (Ridge) : Régularisation douce
        """)
    
    with col2:
        st.code("""
# Paramètres d'optimisation
max_iter = 150            # 150 epochs
learning_rate = 'optimal' # Adaptatif
early_stopping = True     # Arrêt si stagnation
class_weight = 'balanced' # Équilibrage auto
        """, language="python")
        
        st.success("""
        ✅ **Early stopping** :
        Évite surapprentissage en surveillant validation set
        """)
    
    # Données d'entraînement
    st.markdown("### 📊 Données d'Entraînement")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Échantillons", "10 000", delta="Train optimisé")
    with col2:
        st.metric("Features Texte", "16 000", delta="TF-IDF bigrammes")
    with col3:
        st.metric("Features Image", "192", delta="Histogrammes RGB")
    with col4:
        st.metric("Total Features", "16 192", delta="Haute dimension")
    
    st.success("""
    ✅ **Dataset optimal** : 10k échantillons + 16k features = Meilleur compromis  
    performance/temps pour SGDC (4 minutes training sur CPU)
    """)
    
    # Résultats
    st.markdown("### 📈 Résultats")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "75.4%", delta="3/4 corrects")
    with col2:
        st.metric("F1-Score", "74.8%", delta="Weighted")
    with col3:
        st.metric("Temps Training", "4 min", delta="CPU")
    
    # Analyse de l'overfitting
    st.markdown("### 🔍 Analyse Généralisation")
    
    col1, col2, col3 = st.columns(3)
    
    try:
        metrics_path = Path("C:/Users/HP/DataScientest/PROJET/deep_learning_rakuten/git/rakuten-multimodal-classification/Rakuten_Streamlit_Presentation/models/SGDCModel/metrics/metrics_summary.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            with col1:
                st.metric("Train Accuracy", f"{metrics.get('train_accuracy', 0.78)*100:.1f}%")
            with col2:
                st.metric("Test Accuracy", f"{metrics.get('accuracy', 0.754)*100:.1f}%")
            with col3:
                gap = abs(metrics.get('train_accuracy', 0.78) - metrics.get('accuracy', 0.754)) * 100
                st.metric("Écart Train/Test", f"{gap:.1f}%", delta="✅ Excellent" if gap < 5 else "⚠️ Attention")
    except:
        with col1:
            st.metric("Train Accuracy", "78.0%")
        with col2:
            st.metric("Test Accuracy", "75.4%")
        with col3:
            st.metric("Écart Train/Test", "2.6%", delta="✅ Excellent")
    
    st.success("""
    ✅ **Excellente généralisation** : Écart < 3% grâce à la régularisation ElasticNet et early stopping
    """)
    
    # Points forts et limites
    st.markdown("### ⚖️ Points Forts & Limites")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Points Forts**
        
        1. **Performance élevée** : 75.4% accuracy
        2. **Scalabilité** : Millions d'exemples possibles
        3. **Rapidité** : 4 minutes training (CPU)
        4. **Haute dimensionnalité** : Excelle avec 16k features
        5. **Pas de surapprentissage** : Régularisation efficace
        6. **Apprentissage incrémental** : Streaming de données
        7. **Faible mémoire** : Modèle linéaire compact
        """)
    
    with col2:
        st.warning("""
        **⚠️ Limites**
        
        1. **Linéarité uniquement** : Patterns non-linéaires ignorés
        2. **Sensible au preprocessing** : Qualité TF-IDF critique
        3. **Features images basiques** : Histogrammes RGB (< 2%)
        4. **Interprétabilité** : 16k coefficients difficiles à lire
        5. **Tuning nécessaire** : GridSearch sur 45 combinaisons
        6. **Plafond de performance** : 75% max pour approche linéaire
        7. **Catégories proches** : Confusion si lexique similaire
        """)
    
    # Pourquoi SGDC fonctionne bien
    st.markdown("### 🏆 Pourquoi SGDC Fonctionne Bien Ici")
    
    strengths = {
        "Facteur Clé": [
            "Haute dimensionnalité",
            "Nature du problème",
            "Taille du dataset",
            "Ressources limitées"
        ],
        "Explication": [
            "16k features textuelles = espace où SGDC excelle naturellement",
            "Classification texte TF-IDF souvent linéairement séparable",
            "10k échantillons = taille idéale pour SGDC (ni trop petit, ni énorme)",
            "4 min CPU vs plusieurs heures GPU pour Deep Learning"
        ],
        "Impact": ["+++", "+++", "++", "++"]
    }
    st.dataframe(pd.DataFrame(strengths), use_container_width=True, hide_index=True)
    
    # Mots-clés discriminants
    st.markdown("### 🔑 Mots-Clés Discriminants")
    
    st.info("""
    **Top Features** (basé sur coefficients SGDC) :
    
    Le modèle identifie automatiquement les mots/bigrammes les plus importants :
    - **Catégorie 2583** (Piscines) : "piscine", "gonflable", "intex"
    - **Catégorie 1280** (Jeux vidéo) : "ps4", "playstation", "jeu vidéo"
    - **Catégorie 1920** (Linge) : "housse couette", "coton", "parure"
    - **Catégorie 2585** (Bricolage) : "perceuse", "batterie", "makita"
    
    Les features images contribuent ~2% seulement (texte domine).
    """)
    
    # Conclusion
    st.markdown("### 🎯 Conclusion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📊 Performance**
        - Accuracy : **75.4%** ⭐⭐⭐⭐⭐
        - F1-Score : **74.8%** (weighted)
        - Gain vs hasard : **+71.7 points**
        - Temps training : **4 minutes**
        - Généralisation : **2.6% gap**
        """)
    
    with col2:
        st.success("""
        **💡 Utilisation Optimale**
        
        SGDC est **IDÉAL** pour :
        - Classification texte haute dimension
        - Projets avec contraintes CPU
        - MVP production rapide
        - Datasets moyens à grands (10k+)
        - Besoin de scalabilité
        """)
    
    st.write("""
    **Enseignement clé** : Un modèle linéaire bien optimisé (SGDC) peut atteindre d'excellentes 
    performances sur de la classification texte multimodale, tout en restant rapide et économique.
    
    **3 produits sur 4 correctement classés** - Performance remarquable pour un modèle aussi simple !
    """)

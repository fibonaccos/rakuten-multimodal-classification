import streamlit as st
import pandas as pd
import json
from pathlib import Path


def render():
    """Affiche la section Random Forest (1m30 de présentation)"""
    
    # Introduction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Principe de Fonctionnement")
        st.write("""
        **Random Forest** = Ensemble de **50 arbres de décision** qui votent collectivement.
        
        **Concept** :
        - Chaque arbre est entraîné sur un échantillon aléatoire des données
        - Chaque arbre utilise un sous-ensemble aléatoire de features
        - Prédiction finale = vote majoritaire des 50 arbres
        
        **Avantage** : Réduit drastiquement le surapprentissage vs arbre unique
        """)
    
    with col2:
        # Charger métriques
        try:
            metrics_path = Path("C:/Users/HP/DataScientest/PROJET/deep_learning_rakuten/git/rakuten-multimodal-classification/Rakuten_Streamlit_Presentation/models/RandomForest/metrics/metrics_summary.json")
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                
                st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%", delta="Test set")
                st.metric("F1-Score", f"{metrics.get('f1_weighted', 0)*100:.1f}%", delta="Weighted")
                st.metric("Écart Train/Test", f"{metrics.get('overfitting_gap', 0)*100:.1f}%", delta="✅ Excellent")
            else:
                st.metric("Accuracy", "50.8%", delta="Test set")
                st.metric("F1-Score", "48.5%", delta="Weighted")
                st.metric("Écart Train/Test", "4.2%", delta="✅ Excellent")
        except:
            st.metric("Accuracy", "50.8%", delta="Test set")
            st.metric("F1-Score", "48.5%", delta="Weighted")
    
    # Hyperparamètres
    st.markdown("### ⚙️ Configuration Hyperparamètres")
    
    st.write("Optimisés par **GridSearch** pour équilibrer performance et généralisation :")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code("""
# Paramètres de l'ensemble
n_estimators = 50          # 50 arbres
class_weight = 'balanced'  # Équilibrage auto

# Paramètres de chaque arbre
criterion = 'gini'         # Mesure impureté
max_depth = 20            # Profondeur max
        """, language="python")
    
    with col2:
        st.code("""
# Paramètres de régularisation
min_samples_split = 30    # Min pour split
min_samples_leaf = 15     # Min par feuille
max_features = 0.7        # 70% features/split
ccp_alpha = 0.001         # Pruning léger
        """, language="python")
    
    st.info("""
    💡 **Justification** : Ces hyperparamètres résultent d'une optimisation par GridSearch 
    pour trouver le meilleur équilibre entre performance et généralisation.
    """)
    
    # Données d'entraînement
    st.markdown("### 📊 Données d'Entraînement")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Échantillons", "5 000", delta="Réduit pour vitesse")
    with col2:
        st.metric("Features Texte", "10 000", delta="TF-IDF")
    with col3:
        st.metric("Features Image", "192", delta="Histogrammes RGB")
    
    st.warning("""
    ⚠️ **Dataset réduit** : Pour des raisons de temps de calcul, le Random Forest est entraîné 
    sur un échantillon de 5000 produits. Cela impacte la performance finale.
    """)
    
    # Résultats
    st.markdown("### 📈 Résultats")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "50.8%", delta="1/2 corrects")
    with col2:
        st.metric("F1-Score", "48.5%", delta="Weighted")
    with col3:
        st.metric("Temps Training", "30 sec", delta="CPU")
    
    st.info("""
    **Performance modérée** : 50.8% d'accuracy sur le test set. Amélioration significative 
    vs un arbre unique (~40%), mais reste limité pour ce problème haute dimension.
    """)
    
    # Points forts et limites
    st.markdown("### ⚖️ Points Forts & Limites")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Points Forts**
        
        1. **Robustesse** : Moins de surapprentissage qu'un seul arbre
        2. **Interprétabilité** : Feature importance disponible ⭐⭐⭐⭐
        3. **Excellente généralisation** : Gap seulement 4.2%
        4. **Parallélisable** : Arbres indépendants
        5. **Gère bien le bruit** : Vote majoritaire lisse les erreurs
        6. **Rapide** : 30 secondes training
        7. **Patterns non-linéaires** : Peut capturer des relations complexes
        """)
    
    with col2:
        st.error("""
        **❌ Limites**
        
        1. **Haute dimensionnalité** : Souffre avec 10k features
        2. **Performance modérée** : 50.8% accuracy
        3. **Espace de recherche explosif** : 10k features à évaluer
        4. **Complexité** : 50 arbres difficiles à interpréter individuellement
        5. **Mémoire** : Stocke 50 arbres complets
        6. **Scalabilité limitée** : Pas optimal pour millions d'exemples
        7. **Dataset réduit** : 5k échantillons = vocabulaire incomplet
        """)
    
    # Pourquoi performance limitée
    st.markdown("### 🤔 Pourquoi Performance Limitée ?")
    
    reasons = {
        "Facteur Limitant": [
            "Haute dimensionnalité (10k features)",
            "Nature du problème",
            "Dataset réduit (5k)",
            "Type de features"
        ],
        "Explication": [
            "Arbres souffrent du curse of dimensionality - trop de features à évaluer",
            "Classification texte souvent linéairement séparable - arbres cherchent du non-linéaire",
            "Moins de données = vocabulaire incomplet, patterns moins appris",
            "TF-IDF sparse haute dimension inadapté aux arbres de décision"
        ],
        "Impact": ["---", "--", "--", "-"]
    }
    st.dataframe(pd.DataFrame(reasons), use_container_width=True, hide_index=True)
    
    # Feature importance
    st.markdown("### 🔍 Feature Importance")
    
    st.info("""
    **Avantage majeur** : Random Forest identifie automatiquement les mots les plus discriminants.
    
    **Top Features Importantes** (approximatif basé sur TF-IDF) :
    - Mots spécifiques aux catégories (ex: "piscine", "console", "livre")
    - Bigrammes informatifs (ex: "jeu vidéo", "linge maison")
    - Features images contribuent peu (~2% importance totale)
    
    **Utilité** : Permet de comprendre quels mots sont les plus utiles pour la classification.
    """)
    
    # Conclusion
    st.markdown("### 🎯 Conclusion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📊 Performance**
        - Accuracy : **50.8%**
        - F1-Score : **48.5%** (weighted)
        - Gain vs arbre unique : **~+10 pts**
        - Généralisation : **4.2% gap**
        - Temps : **30 secondes**
        """)
    
    with col2:
        st.success("""
        **💡 Utilisation Recommandée**
        
        Random Forest est **pertinent** pour :
        - Projets nécessitant interprétabilité
        - Features < 1000 dimensions
        - Budgets computationnels limités
        - Baseline rapide avant deep learning
        - Analyse feature importance
        """)
    
    st.write("""
    **Enseignement clé** : Random Forest est un excellent modèle pour de nombreux problèmes, 
    mais souffre face à la haute dimensionnalité textuelle (10k features). Son point fort 
    reste l'**interprétabilité** et la **robustesse** avec une excellente généralisation (pas d'overfitting).
    """)

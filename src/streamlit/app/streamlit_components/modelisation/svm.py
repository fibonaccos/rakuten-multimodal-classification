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
        **SVM (Support Vector Machine)** = Séparateur à Vaste Marge.
        
        **Concept** :
        - Cherche l'hyperplan qui maximise la marge entre les différentes classes.
        - Se base uniquement sur les points les plus difficiles à classer : les vecteurs de support.
        
        
        **Particularité** : Performant avec des ensembles de données non structurés.        """)
    
    with col2:
        
        st.metric("Accuracy", "72%")
        st.metric("F1-Score", "70%")
    
    # Hyperparamètres optimisés
    st.markdown("### ⚙️ Configuration Hyperparamètres")
    
    st.write("Optimisés par **GridSearch exhaustif** sur 45 combinaisons :")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code("""
# Fonction de perte et régularisation
loss = 'hinge'         
penalty = 'l2'     
alpha = 0.0001

# Paramètres d'optimisation
max_iter = 1000          
        """, language="python")
        
        st.info("""
        💡 **loss = 'hinge** est la fonction de perte utilisée pour l'entrainement des modèles SVM
        """)
    
    
    # Données d'entraînement
    st.markdown("### 📊 Données d'Entraînement")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Échantillons", "68 000", delta="0.8 du dataset")
    with col2:
        st.metric("Features Texte", "10 000", delta="TF-IDF Unigrammes + Bigrammes, limité par max_features")
    with col3:
        st.metric("Total Features", "10000")
    
    # Résultats
    st.markdown("### 📈 Résultats")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "72%", delta="3/4 corrects")
    with col2:
        st.metric("F1-Score", "70%")
    with col3:
        st.metric("Temps Training", "2 min", delta="CPU")
    
    
    # Points forts et limites
    st.markdown("### ⚖️ Points Forts & Limites")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Points Forts**
        
        1. **Vitesse** : moins de 2 minutes
        2. **Robustesse** : La perte Hinge est moins sensible aux outliers que la perte logistique.
        3. **Simplicité** : Pas de features images complexes, pipeline purement textuel.
        """)
    
    with col2:
        st.warning("""
        **⚠️ Limites**
        
        1. **Moins précis** : Perte d'accuracy comparé à d'autres modèles
        2. **Absence de probabilité** : SVM ne donne pas de score de confiance 
        3. **Unimodal** : On ignore les images
        """)
    
    # Pourquoi SGDC fonctionne bien
    st.markdown("### 🏆 Pourquoi SVM Fonctionne Bien Ici")
    
    st.write("Même simpliste, le SVM linéaire est extrêmement efficace sur le texte (haute dimension, données éparses). Il se concentre sur les mots clés discriminants qui définissent la frontière de chaque catégorie.")
    
    
    
    # Conclusion
    st.markdown("### 🎯 Conclusion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📊 Performance**
        - Accuracy : **72%** 
        - F1-Score : **70%** 
        - Temps training : **2 minutes**
   
        """)
    
        
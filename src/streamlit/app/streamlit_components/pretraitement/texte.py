import streamlit as st
import pandas as pd
import json
from pathlib import Path

def render():
    """Affiche la section SGDClassifier (1m30 de présentation)"""
    
    # Introduction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Présentation des différentes étapes")
        st.write("""
       
        - Nettoyage Syntaxique
        - Gestion des différents languages
        - Traitement des descriptions vides
        - Vectorisation
                """)
    
    # Hyperparamètres optimisés
    st.markdown("### 🧹 Nettoyage Syntaxique")

    st.write(""" La première étape consiste à standardiser la forme du texte
             - Encodage & HTML : Suppression des entités HTML (ex: &amp;, <div>) et décodage des caractères spéciaux via html.unescape.
             - Normalisation Unicode : Transformation des caractères accentués en leur équivalent ASCII (ex: "été" -> "ete") via la norme NFKD
             - Casse : Conversion de l'ensemble du texte en minuscules pour éviter que "Table" et "table" soient considérés comme deux mots différents.
""")
    
    st.markdown("### ⚙️ Filtrage des expressions régulières (Regex) ")

    st.write(""" Pour réduire la dimensionnalité, nous appliquons un filtre strict à savoir la conservation exclusive des lettres (a-z)
""")
    st.code("""
            text = re.sub(r'<[^>]+>', ' ', text)       # HTML
            text = re.sub(r'\b\w*\d\w*\b', ' ', text)  # Mots avec chiffres 
            text = re.sub(r'[^a-z\s]', ' ', text)      # Lettres a-z uniquement
            text = re.sub(r'\s+', ' ', text).strip()   # Espaces        
        """, language="python")
    
    st.markdown("### ⛔  Suppression des STOPWORDS ")

    st.write(""" Nous utilisons une liste d'exclusion personnalisée (basée sur NLTK et enrichie manuellement) pour retirer les mots très fréquents mais peu informatifs
""")
    st.markdown("### ✅️ Lemmatisation ")

    st.write(""" Plutôt que la racinisation (stemming) qui coupe brutalement les mots, nous avons opté pour la lemmatisation via la librairie spaCy (modèle fr_core_news_sm).
             Le principe est de transformer chaque mot en sa forme canonique. 
""")
    st.markdown("### ⬆️ Gestion des Données Manquantes & Augmentation ")

    st.write(""" - **Valeurs nulles** : Les descriptions manquantes (NaN) ne sont pas supprimées mais remplacées par des chaînes vides ou fusionnées avec la colonne designation (titre du produit) pour maximiser l'information disponible.
             - Traduction : A l'aide de Spacy, on à décidé de traduire les 5 langues les plus représentées dans notre dataset
""")
    
    st.markdown("### 📈 Vectorisation ")

    st.write(""" Nous avons opté pour la méthode TF-IDF (Term Frequency - Inverse Document Frequency), qui est une approche statistique robuste privilégiant les mots "porteurs de sens" spécifiques à chaque catégorie.
             Nous avons utilisé TfidfVectorizer de Scikit-Learn avec les hyperparamètres suivants, optimisés pour maximiser le ratio performance/mémoire :
             """)
    st.code("""
            vectorizer = TfidfVectorizer(ngram_range=(1,2), #permet d'analyser les mots et les paires de mots 
            min_df=3, # Un mot (ou bigramme) doit apparaître dans au moins 3 produits différents pour être conservé.
            max_features = 10 000 # Nous ne gardons que les 10 000 mots les plus fréquents du corpus.)
""", language="python")
    

    
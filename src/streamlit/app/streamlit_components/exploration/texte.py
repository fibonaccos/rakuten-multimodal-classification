import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@st.cache_data
def load_data():
    """Charge les données Y_train"""
    try:
        data_path = PROJECT_ROOT / "data" / "Y_train_CVw08PX.csv"
        if data_path.exists():
            return pd.read_csv(data_path)
    except Exception:
        pass
    return None


def render():
    """Affiche l'exploration des données TEXTE (1m40 de présentation)"""
    
    st.markdown("## 📝 Exploration des Données Textuelles")
    
    # Structure du dataset
    st.markdown("### 📊 Structure du Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **X_train.csv : 84 916 lignes × 5 colonnes**
        
        - `designation` : Titres des produits
        - `description` : Descriptions détaillées
        - `productid` : ID unique produit
        - `imageid` : ID unique image
        """)
        
        st.success("""
        ✅ **4 colonnes utiles** exploitables
        
        ✅ **Unicité** : Chaque produit unique lié à une image unique
        → Chaque produit est unique dans le dataset
        """)
    
    with col2:
        st.metric("Produits", "84 916", delta="Total dataset")
        st.metric("Variables Texte", "2", delta="designation + description")
        st.metric("IDs Uniques", "100%", delta="productid & imageid")
    
    st.markdown("---")
    
    # Catégories
    st.markdown("### 🏷️ Analyse des Catégories (Y)")
    
    st.write("""
    **Y_train.csv** : 84 916 lignes × 2 colonnes (Unnamed: 0 + prdtypecode)
    
    La colonne `prdtypecode` contient **27 catégories distinctes** à prédire.
    """)
    
    # Charger les données
    y_data = load_data()
    
    # Distribution des catégories
    st.markdown("#### Distribution des Catégories")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Catégories", "27", delta="Classes à prédire")
    with col2:
        st.metric("Classe Max", "2583", delta="12.02% du dataset")
    with col3:
        st.metric("Déséquilibre", "~1:50", delta="Entre min et max")
    
    st.error("""
    **⚠️ Déséquilibre Inter-Classes Majeur**
    
    - **Catégorie 2583** (sur-représentée) : 10 209 produits (12.02%)
    - **Catégories sous-représentées** : 2905, 60, 2220, 1301, 1940, 1180
    
    → Nécessite ré-échantillonnage : sous-échantillonnage de 2583 + sur-échantillonnage minoritaires
    """)
    
    # Champs lexicaux
    st.markdown("#### Champs Lexicaux et Corrélations")
    
    st.write("""
    **Analyse par nuages de mots** (wordcloud) : Construction sur designation + description concaténées 
    pour révéler les univers lexicaux distincts de chaque catégorie.
    
    **Matrice de corrélation lexicale** (heatmap) : Calcul des corrélations entre catégories pour 
    identifier les champs lexicaux rapprochés.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Catégories bien distinctes**
        
        Champs lexicaux clairement identifiables :
        - **2583** : "piscine", "gonflable", "eau"
        - **1280** : "ps4", "jeu", "console"
        - **1920** : "housse", "coton", "lit"
        - **2585** : "outil", "vis", "électrique"
        """)
    
    with col2:
        st.warning("""
        **⚠️ Catégories à champs rapprochés**
        
        Corrélation lexicale élevée (heatmap) :
        - **(1280, 1281)** : Jeux généraux vs PC
        - **(50, 2462)** : Accessoires gaming
        - **(1280, 1302)** : Jeux vs Accessoires
        - **(1560, 2582)** : Mobilier intérieur/extérieur
        
        → Catégorie **1280** particulièrement corrélée
        """)
    
    st.info("""
    💡 **Outils utilisés** : Nuages de mots (wordcloud) + Matrice de corrélation (heatmap) pour 
    identifier visuellement les similitudes et différences lexicales entre catégories.
    """)
    
    # === VISUALISATIONS STATIQUES ===
    with st.expander("📊 Voir les visualisations (Distribution + Nuages de mots + Langues)"):
        try:
            # Chemins vers les images statiques
            assets_path = Path(__file__).parent.parent.parent / "assets"
            
            # 1. Distribution des classes
            st.markdown("##### 📊 Distribution des Catégories")
            class_dist_path = assets_path / "class_distribution.png"
            if class_dist_path.exists():
                st.image(str(class_dist_path), use_container_width=True)
            else:
                st.info("Graphique non disponible")
            
            st.markdown("---")
            
            # 2. Nuages de mots
            st.markdown("##### ☁️ Nuages de Mots pour Quelques Catégories")
            wordcloud_path = assets_path / "wordclouds.png"
            if wordcloud_path.exists():
                st.image(str(wordcloud_path), use_container_width=True)
            else:
                st.info("Nuages de mots non disponibles")
            
            st.markdown("---")
            
            # 3. Distribution des langues
            st.markdown("##### 🌍 Distribution des Langues")
            lang_dist_path = assets_path / "language_distribution.png"
            if lang_dist_path.exists():
                st.image(str(lang_dist_path), use_container_width=True)
            else:
                st.info("Graphique non disponible")
                    
        except Exception as e:
            st.error(f"Erreur lors du chargement des visualisations : {e}")
    
    st.markdown("---")
    
    # Données textuelles
    st.markdown("### 📝 Variables Textuelles : designation & description")
    
    # Valeurs manquantes
    st.markdown("#### 🚨 Valeurs Manquantes")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.error("""
        **35.09% de valeurs manquantes** dans `description`
        
        **Catégories les plus touchées** (taux > 60%) :
        - **2403** : 88.5%
        - **2462** : 81.2%
        - **2280** : 77.3%
        - **1160** : 69.8%
        - **10** : 65.4%
        - **1180** : 63.7%
        - **40** : 61.5%
        - **1140** : 60.1%
        
        **Explications** :
        - Description = champ non obligatoire sur Rakuten
        - Vendeurs intègrent parfois description dans le titre
        - Qualité variable selon catégories
        """)
    
    with col2:
        st.success("""
        **✅ Catégories propres**
        
        Taux < 5% de manquants :
        - **2905** : 0% (parfait!)
        - **1920** : faible
        - **1560** : faible
        - **2582** : faible
        
        Données de qualité optimale
        """)
    
    # Analyse textuelle
    st.markdown("#### 🔍 Analyse Textuelle - Qualité des Données")
    
    st.write("""
    **Problèmes identifiés dans les variables textuelles :**
    """)
    
    quality_data = {
        "Problème": [
            "Code HTML résiduel",
            "Caractères spéciaux",
            "URLs et emails",
            "Langues étrangères"
        ],
        "Variable": [
            "description (important)",
            "designation (26) / description (47)",
            "Présents sporadiquement",
            "~15% non-français"
        ],
        "Impact": [
            "Pollue le texte et fausse les longueurs",
            "Incompatibilités avec modèles",
            "Information non pertinente",
            "Nécessite encodage multilingue"
        ]
    }
    st.dataframe(pd.DataFrame(quality_data), use_container_width=True, hide_index=True)
    
    st.info("""
    💡 **Constat** : Nettoyage poussé nécessaire, notamment pour vectorisation TF-IDF 
    (tokenisation basée sur mots). Les transformers multilingues seraient avantageux pour 
    gérer les langues étrangères.
    """)
    
    # Distribution des langues
    st.markdown("#### 🌍 Langues Détectées")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Français", "~85%", delta="Majoritaire")
    with col2:
        st.metric("Anglais", "~8%", delta="Produits internationaux")
    with col3:
        st.metric("Autres", "~7%", delta="Espagnol, Italien, etc.")
    
    # Distribution des longueurs
    st.markdown("#### 📏 Distribution des Longueurs de Texte")
    
    st.write("""
    **Analyse variable `designation` :**
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Avec description présente**")
        length_with = {
            "Métrique": ["Moyenne", "Écart-type", "Observation"],
            "Valeur": ["73.67 car.", "29.81", "Homogène et détaillé"]
        }
        st.table(pd.DataFrame(length_with))
        
        st.success("""
        ✅ Titres plus longs et homogènes quand description présente
        → Effort du vendeur, informations précises
        """)
    
    with col2:
        st.markdown("**Sans description**")
        length_without = {
            "Métrique": ["Moyenne", "Écart-type", "Observation"],
            "Valeur": ["63.67 car.", "46.36", "Plus court et hétérogène"]
        }
        st.table(pd.DataFrame(length_without))
        
        st.warning("""
        ⚠️ **Pic caractéristique à 240-250 car.**
        → Limite système : description intégrée dans titre
        → Vendeurs bloqués par taille maximale
        """)
    
    st.info("""
    💡 **Découverte importante** : 
    - Longueur maximale de `designation` : **250 caractères**
    - Pic à 240-250 car. quand pas de description → Vendeurs utilisent le titre comme description
    - Stratégie possible : Si longueur(titre) > 150 car. ET pas de description → Considérer titre comme description
    """)
    
    st.markdown("---")
    
    # Synthèse
    st.markdown("### 🎯 Synthèse de l'Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Points Forts**
        
        1. Volume suffisant : 84 916 produits
        2. Unicité garantie (productid/imageid)
        3. Images uniformes (500×500 RGB)
        4. Champs lexicaux distincts majoritairement
        5. 27 catégories = nombre modéré
        """)
    
    with col2:
        st.error("""
        **⚠️ Défis Identifiés**
        
        1. **35% descriptions manquantes**
        2. **Déséquilibre 1:50** (catégorie 2583)
        3. Code HTML, caractères spéciaux
        4. 15% textes en langues étrangères
        5. Catégories lexicalement proches
        """)
    
    st.write("""
    **→ Ces observations guideront les choix de preprocessing et modélisation.**
    """)

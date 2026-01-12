import streamlit as st
import pandas as pd


def render():
    """Affiche l'exploration des données TEXTE (1m40 de présentation)"""
    
    st.markdown("## 📝 Exploration des Données Textuelles (X)")
    
    # Structure du dataset
    st.markdown("### 📊 Structure du Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **X_train.csv : 84 916 lignes × 5 colonnes**
        
        - `Unnamed: 0` : Indices (inutile)
        - `designation` : Titres des produits
        - `description` : Descriptions détaillées
        - `productid` : ID unique produit
        - `imageid` : ID unique image
        """)
        
        st.success("""
        ✅ **4 colonnes utiles** : designation, description, productid, imageid
        
        ✅ **Unicité** : Valeurs uniques dans productid et imageid
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
    
    # === VISUALISATIONS INTERACTIVES ===
    with st.expander("📊 Voir les visualisations (Distribution + Nuages de mots + Langues)"):
        try:
            # Import des fonctions du collègue
            import matplotlib.pyplot as plt
            import seaborn as sns
            from wordcloud import WordCloud
            import re
            from bs4 import BeautifulSoup
            import nltk
            from nltk.corpus import stopwords
            
            # Tentative de chargement des données
            try:
                # Chemins possibles
                data_paths = [
                    "C:/Users/Peeta/Desktop/Projet/rakuten-multimodal-classification/data/raw/X_train_update.csv",
                    "data/raw/X_train_update.csv",
                    "../../../../../data/raw/X_train_update.csv"
                ]
                
                y_paths = [
                    "C:/Users/Peeta/Desktop/Projet/rakuten-multimodal-classification/data/raw/Y_train_CVw08PX.csv",
                    "data/raw/Y_train_CVw08PX.csv",
                    "../../../../../data/raw/Y_train_CVw08PX.csv"
                ]
                
                df_x = None
                df_y = None
                
                for x_path, y_path in zip(data_paths, y_paths):
                    try:
                        df_x = pd.read_csv(x_path)
                        df_y = pd.read_csv(y_path)
                        break
                    except:
                        continue
                
                if df_x is None:
                    st.warning("⚠️ Données brutes non disponibles. Visualisations désactivées.")
                else:
                    df = df_x.copy()
                    df['prdtypecode'] = df_y['prdtypecode']
                    
                    # 1. Distribution des classes
                    st.markdown("##### 📊 Distribution des Catégories")
                    
                    type_counts = df['prdtypecode'].value_counts()
                    fig1, ax1 = plt.subplots(figsize=(12, 6))
                    type_counts.plot(kind='bar', color=sns.color_palette("viridis", len(type_counts)), ax=ax1)
                    ax1.set_title('Distribution des Types de Produits', fontsize=16, fontweight='bold')
                    ax1.set_xlabel('Type de Produit', fontsize=14)
                    ax1.set_ylabel('Nombre d\'Occurrences', fontsize=14)
                    plt.xticks(rotation=45, fontsize=12)
                    
                    for index, value in enumerate(type_counts):
                        ax1.text(index, value, str(value), ha='center', va='bottom', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig1)
                    plt.close()
                    
                    # 2. Nuages de mots
                    st.markdown("##### ☁️ Nuages de Mots par Catégorie")
                    
                    # Fonction de nettoyage
                    def get_stopwords():
                        try:
                            return set(stopwords.words('french'))
                        except:
                            return set()
                    
                    FINAL_STOPWORDS = get_stopwords()
                    
                    def throw_html_elem(text):
                        if not isinstance(text, str): return ""
                        try:
                            return BeautifulSoup(text, "html.parser").get_text(separator=" ")
                        except:
                            return text
                    
                    def basic_clean(text):
                        if not isinstance(text, str): return ""
                        text = text.lower()
                        text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", " ", text)
                        text = re.sub(r"\s+", " ", text)
                        words = text.split()
                        words = [w for w in words if w not in FINAL_STOPWORDS and len(w) > 2]
                        return " ".join(words)
                    
                    # Préparer les données
                    df['lexical_field'] = df['designation'].fillna('') + ' ' + df['description'].fillna('')
                    df['lexical_field'] = df['lexical_field'].apply(lambda s: basic_clean(throw_html_elem(s)))
                    text_by_class = df.groupby('prdtypecode')['lexical_field'].apply(lambda s: ' '.join(s))
                    
                    available_classes = sorted(text_by_class.index.unique())
                    selected_class = st.selectbox("Choisir une classe :", available_classes, index=0)
                    
                    if selected_class:
                        text_content = text_by_class[selected_class]
                        
                        if text_content and len(text_content.strip()) > 0:
                            wc = WordCloud(width=800, height=400, background_color='white', colormap='cividis').generate(text_content)
                            
                            fig_wc, ax_wc = plt.subplots(figsize=(10, 6))
                            ax_wc.imshow(wc, interpolation='bilinear')
                            ax_wc.set_title(f'Nuage de mots - Classe : {selected_class}', fontsize=16)
                            ax_wc.axis('off')
                            
                            st.pyplot(fig_wc)
                            plt.close()
                        else:
                            st.warning(f"Pas assez de mots clés pour la classe {selected_class}")
                    
                    # 3. Distribution des langues
                    st.markdown("##### 🌍 Distribution des Langues")
                    
                    donnees_langues = {
                        "Français": 27000,
                        "Anglais": 7600,
                        "Italien": 5000,
                        "Unknown": 3000,
                        "Roumain": 2500,
                        "Espagnol": 1200
                    }
                    
                    st.bar_chart(donnees_langues)
                    
                    st.caption("""
                    📌 **Note** : Distribution approximative basée sur détection automatique avec `langdetect`.
                    Le français domine (~70%), mais présence significative de langues étrangères (~15-20%).
                    """)
                    
            except Exception as e:
                st.error(f"Erreur lors du chargement des visualisations : {e}")
                
        except ImportError as e:
            st.warning(f"⚠️ Bibliothèques manquantes pour les visualisations : {e}")
    
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

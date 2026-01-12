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

# Chemins
ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"


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
    try:
        import nltk
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    except ImportError:
        pass


def get_all_stopwords():
    """Récupère tous les stopwords français étendus"""
    try:
        from nltk.corpus import stopwords
        stopwds = set(stopwords.words('french'))
    except:
        stopwds = set()
    
    added_stopwds = {
        'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'me', 'te', 'se', 'moi', 'toi', 'soi', 'leur', 'lui',
        'en', 'y', 'ce', 'cela', 'ça', 'ceci', 'celui', 'celle', 'ceux', 'celles','mon', 'ton', 'son', 'notre', 'votre',
        'leur', 'mes', 'tes', 'ses', 'nos', 'vos', 'leurs', 'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'au', 'aux',
        'ce', 'ces', 'cet', 'cette', 'être', 'avoir', 'faire', 'aller', 'venir', 'pouvoir', 'devoir', 'savoir', 'dire',
        'voir', 'mettre', 'prendre', 'donner', 'vouloir', 'falloir', 'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car', 'par',
        'pour', 'dans', 'sur', 'sous', 'avec', 'sans', 'entre', 'chez', 'vers', 'selon', 'depuis', 'pendant', 'autour',
        'après', 'avant', 'si', 'quand', 'comme', 'bien', 'très', 'trop', 'peu', 'aussi', 'encore', 'déjà', 'toujours',
        'jamais', 'parfois', 'souvent', 'moins', 'plus', 'autant', 'alors', 'ensuite', 'également', 'tout', 'tous',
        'toutes', 'chaque', 'aucun', 'certaines', 'certains', 'plusieurs', 'autre', 'autres', 'même', 'tel', 'tels',
        'tellement', 'chose', 'truc', 'cas', 'façon', 'manière', 'genre', 'type'
    }
    stopwds.update(added_stopwds)
    return stopwds


def throw_html_elem(text: str) -> str:
    """Supprime les éléments HTML"""
    if not isinstance(text, str): 
        return ""
    try:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except:
        return text


def basic_clean(text: str, stopwords_set) -> str:
    """Nettoie le texte pour wordcloud"""
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", " ", text) 
    text = re.sub(r"\s+", " ", text)  
    words = text.split()
    words = [w for w in words if w not in stopwords_set and len(w) > 2]
    return " ".join(words)


@st.cache_data
def process_text_for_wordclouds(df):
    """Prépare les textes par catégorie pour wordclouds"""
    stopwords_set = get_all_stopwords()
    data = df.copy()
    
    data['lexical_field'] = data['designation'].fillna('') + ' ' + data['description'].fillna('')
    data['lexical_field'] = data['lexical_field'].apply(lambda s: basic_clean(throw_html_elem(s), stopwords_set))
    
    grouped_text = data.groupby('prdtypecode')['lexical_field'].apply(lambda s: ' '.join(s))
    
    return grouped_text


def generate_interactive_wordclouds():
    """Génère les nuages de mots interactifs avec boutons"""
    try:
        from wordcloud import WordCloud
    except ImportError:
        st.warning("⚠️ Module wordcloud non disponible")
        return
    
    download_nltk_resources()
    
    # Charger les données
    df = load_full_data()
    
    if df is None:
        st.info("📁 Données brutes non disponibles localement - nuages de mots désactivés")
        return
    
    # Préparer les textes
    with st.spinner("Traitement des textes..."):
        text_by_class = process_text_for_wordclouds(df)
    
    available_classes = sorted(text_by_class.index.unique())
    
    # Mapping des catégories
    category_names = {
        10: "Livres/Médias", 40: "Jeux vidéo/Consoles", 50: "Accessoires gaming",
        60: "Consoles", 1140: "Figurines", 1160: "Cartes collectibles",
        1180: "Figurines", 1280: "Jeux (général)", 1281: "Jeux PC",
        1300: "Déco intérieure", 1301: "Déco extérieure", 1302: "Accessoires consoles",
        1320: "Mobilier intérieur", 1560: "Mobilier extérieur", 1920: "Linge de maison",
        1940: "Literie/Ameublement", 2060: "Déco murale", 2220: "Équipement animaux",
        2280: "Magazines", 2403: "Livres (autre type)", 2462: "Jeux/Consoles retro",
        2522: "Papeterie", 2582: "Mobilier extérieur", 2583: "Piscines",
        2585: "Bricolage", 2705: "Livres", 2905: "Jeux de société"
    }
    
    st.write("**Sélectionnez une catégorie pour visualiser son nuage de mots :**")
    
    # Créer des colonnes pour les boutons (5 boutons par ligne)
    cols_per_row = 5
    rows = [available_classes[i:i+cols_per_row] for i in range(0, len(available_classes), cols_per_row)]
    
    # Session state pour stocker la sélection
    if 'selected_wordcloud_class' not in st.session_state:
        st.session_state.selected_wordcloud_class = available_classes[0]
    
    # Afficher les boutons
    for row in rows:
        cols = st.columns(cols_per_row)
        for idx, cat_code in enumerate(row):
            cat_name = category_names.get(cat_code, f"Cat {cat_code}")
            with cols[idx]:
                if st.button(f"{cat_code}\n{cat_name}", key=f"wc_{cat_code}", use_container_width=True):
                    st.session_state.selected_wordcloud_class = cat_code
    
    # Afficher le wordcloud pré-généré pour la catégorie sélectionnée
    selected_class = st.session_state.selected_wordcloud_class
    st.markdown(f"### Catégorie : **{selected_class}** - *{category_names.get(selected_class, 'N/A')}*")
    
    # Charger l'image pré-générée
    wordcloud_path = ASSETS_DIR / "wordclouds" / f"wordcloud_{selected_class}.png"
    
    if wordcloud_path.exists():
        st.image(str(wordcloud_path), use_container_width=True)
    else:
        st.warning(f"Nuage de mots non disponible pour la catégorie {selected_class}")


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
    
    # === VISUALISATIONS ===
    with st.expander("📊 Voir les visualisations (Distribution + Nuages de mots)"):
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
            
            # 2. Nuages de mots INTERACTIFS
            st.markdown("##### ☁️ Nuages de Mots par Catégorie (Interactif)")
            generate_interactive_wordclouds()
                    
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
    
    # Graphique distribution des langues
    try:
        assets_path = Path(__file__).parent.parent.parent / "assets"
        lang_dist_path = assets_path / "language_distribution.png"
        if lang_dist_path.exists():
            st.image(str(lang_dist_path), use_container_width=True)
    except Exception:
        pass
    
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

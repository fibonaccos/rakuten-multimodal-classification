import streamlit as st
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException

st.title("Introduction")

@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_resources()

def get_all_stopwords():
    stopwds = set(stopwords.words('french'))
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

FINAL_STOPWORDS = get_all_stopwords()

def throw_html_elem(text: str) -> str:
    if not isinstance(text, str): return ""
    try:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except:
        return text

def basic_clean(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", " ", text) 
    text = re.sub(r"\s+", " ", text)  
    words = text.split()
    words = [w for w in words if w not in FINAL_STOPWORDS and len(w) > 2]
    return " ".join(words)

@st.cache_data
def load_data():
    x_path = r"C:/Users/mazoyerro/STREAMLITSL/streamlit/rakuten-multimodal-classification/data/raw/X_train_update.csv"
    y_path = r"C:/Users/mazoyerro/STREAMLITSL/streamlit/rakuten-multimodal-classification/data/raw/Y_train_CVw08PX.csv"

    df_x = pd.read_csv(x_path)
    df_y = pd.read_csv(y_path)

    df = df_x.copy()
    df['prdtypecode'] = df_y['prdtypecode']
    
    return df

def process_text_for_wordclouds(df):
    """
    Applique le nettoyage et le groupement par classe.
    Retourne une Série avec le texte concaténé par code produit.
    """
    data = df.copy()
    
    data['lexical_field'] = data['designation'].fillna('') + ' ' + data['description'].fillna('')

    data['lexical_field'] = data['lexical_field'].apply(lambda s: basic_clean(throw_html_elem(s)))
    
    grouped_text = data.groupby('prdtypecode')['lexical_field'].apply(lambda s: ' '.join(s))
    
    return grouped_text

def analyze_languages(df):
    """
    Détecte la langue de chaque description.
    Retourne une Série avec les comptes par langue.
    """
    def safe_detect(text):
        try:
            # On ignore les textes trop courts (< 3 chars) ou vides
            if not isinstance(text, str) or len(text.strip()) < 3:
                return 'unknown'
            return detect(text)
        except LangDetectException:
            return 'unknown'

    # On travaille sur une copie pour ne pas impacter le dataframe global
    descriptions = df['description'].fillna("").astype(str)
    
    # Application de la détection (peut être long)
    languages = descriptions.apply(safe_detect)
    
    return languages.value_counts()

try:
    df = load_data()
except FileNotFoundError:
    st.error("Fichiers CSV introuvables. Vérifiez les chemins dans load_data().")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "Contexte & objectifs",
    "Données & hypothèses",
    "Exploration"
])

with tab1:
    st.header("Contexte & objectifs")
    st.subheader("Contexte")

    st.subheader("Objectifs")

with tab2:
    st.header("Données & hypothèses")
    st.subheader("Données")

    st.subheader("Hypothèses")

with tab3:
    st.subheader("Données textuelles")
    
    st.markdown("""
    **Analyse de la distribution des classes :**
                
    Le graphique ci-dessous montre la répartition des produits par code catégorie (`prdtypecode`). 
    On observe un déséquilibre important entre les classes, ce qui est un point d'attention crucial pour l'entraînement des modèles.
    """)

    if 'prdtypecode' in df.columns:
        type_counts = df['prdtypecode'].value_counts()

        sns.set(style="white")

        fig = plt.figure(figsize=(12, 6))
        
        bar_plot = type_counts.plot(kind='bar', color=sns.color_palette("viridis", len(type_counts)))

        plt.title('Distribution des Types de Produits', fontsize=16, fontweight='bold')
        plt.xlabel('Type de Produit', fontsize=14)
        plt.ylabel('Nombre d\'Occurrences', fontsize=14)

        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)

        for index, value in enumerate(type_counts):
            plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        st.pyplot(fig)
    else:
        st.error("La colonne 'prdtypecode' est introuvable dans le DataFrame.")

    st.markdown("""
    **Analyse sémantique par classe :**
                
    Le graphique ci-dessous montre la répartition et l'importance des mots définissant une classe à l'aide d'un nuage de mot. 
    """)

    if 'prdtypecode' in df.columns:
        with st.spinner("Traitement des textes et génération des champs lexicaux..."):
            text_by_class = process_text_for_wordclouds(df)

        available_classes = sorted(text_by_class.index.unique())
        
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_class = st.selectbox(
                "Choisir une classe :", 
                available_classes,
                index=0
            )
        
        with col2:
            if selected_class:
                text_content = text_by_class[selected_class]
                
                if text_content and len(text_content.strip()) > 0:
                    wc = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white', 
                        colormap='cividis'
                    ).generate(text_content)
                    
                    # Affichage
                    fig_wc, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.set_title(f'Nuage de mots - Classe : {selected_class}', fontsize=16)
                    ax.axis('off')
                    
                    st.pyplot(fig_wc)
                else:
                    st.warning(f"Pas assez de mots clés trouvés pour la classe {selected_class} après nettoyage.")
    else:
        st.error("La colonne 'prdtypecode' est requise pour cette analyse.")

    st.markdown("""
    **Analyses avancées :**
                
    à completer 
    """)
    
    df['designation_length'] = df['designation'].fillna("").astype(str).apply(len)
    df['missing_description'] = df['description'].isna().astype(int)

    for_agg = {
        'label_count': pd.NamedAgg(column='prdtypecode', aggfunc='count'),
        'missing_count': pd.NamedAgg(column='missing_description', aggfunc='sum'),
        'avg_designation_length': pd.NamedAgg(column='designation_length', aggfunc='mean')
    }
    
    labels_aggs = df.groupby('prdtypecode').agg(**for_agg)

    labels_aggs['label_freq'] = labels_aggs['label_count'] / df.shape[0]
    labels_aggs['missing_ratio'] = labels_aggs['missing_count'] / labels_aggs['label_count']

    sorted_labels_freqs = labels_aggs.sort_values(by='label_freq', ascending=False)
    x = sorted_labels_freqs.index.astype(str)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))

    sns.barplot(
        x=x, 
        y=sorted_labels_freqs['label_freq'], 
        hue=sorted_labels_freqs['missing_ratio'], 
        palette='Reds', 
        edgecolor='black', 
        zorder=2, 
        alpha=0.9, 
        ax=ax[0],
        dodge=False
    )
    
    ax[0].set_title('Distribution des classes (Couleur = Taux de descriptions manquantes)', fontsize=14)
    ax[0].set_xlabel('Code Produit (Labels)', fontsize=12)
    ax[0].set_ylabel('Fréquence', fontsize=12)
    ax[0].grid(visible=True, linestyle='--', axis='y', zorder=0)
    
    if ax[0].get_legend():
        ax[0].legend().remove()

    sm1 = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=sorted_labels_freqs['missing_ratio'].min(), vmax=sorted_labels_freqs['missing_ratio'].max()))
    sm1.set_array([])
    cbar1 = plt.colorbar(sm1, ax=ax[0])
    cbar1.set_label('Ratio Description Manquante')

    sns.barplot(
        x=x, 
        y=sorted_labels_freqs['avg_designation_length'], 
        hue=sorted_labels_freqs['missing_ratio'], 
        palette='Reds', 
        edgecolor='black', 
        zorder=2, 
        ax=ax[1],
        dodge=False
    )
    
    ax[1].set_title('Longueur moyenne des désignations par classe', fontsize=14)
    ax[1].set_xlabel('Code Produit (Labels)', fontsize=12)
    ax[1].set_ylabel('Longueur moyenne (caractères)', fontsize=12)
    ax[1].grid(visible=True, linestyle='--', axis='y', zorder=0)
    
    if ax[1].get_legend():
        ax[1].legend().remove()

    sm2 = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=sorted_labels_freqs['missing_ratio'].min(), vmax=sorted_labels_freqs['missing_ratio'].max()))
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=ax[1])
    cbar2.set_label('Ratio Description Manquante')

    plt.tight_layout()
    
    st.pyplot(fig)

    st.markdown("""
    **Analyses  :**
                
    à completer 
    """)

    st.markdown("""
    **Analyse linguistique :**
                
    à completer 
    """)

    








    st.divider()

    st.subheader("Données imagées")

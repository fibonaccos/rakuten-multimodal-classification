"""
Script de prétraitement de données textuelles pour la classification de produits Rakuten.

Ce script effectue les étapes suivantes :
1. Chargement des fichiers X_train et Y_train
2. Nettoyage des descriptions produits :
   - Minuscule
   - Suppression des accents
   - Nettoyage HTML et ponctuation
   - Suppression des stopwords personnalisés
   - Lemmatisation en français avec spaCy
3. Vectorisation TF-IDF des textes nettoyés
4. Affichage d’un aperçu de la matrice TF-IDF pour vérification

Classes et fonctions clés :
- NettoyeurTexte : transformateur personnalisé scikit-learn pour nettoyage + lemmatisation
- remove_accents : fonction utilitaire pour supprimer les accents
- lemmatize : fonction pour extraire les lemmes pertinents avec spaCy

Utilisation :
- Ce script est conçu pour être intégré dans un pipeline scikit-learn complet.
- Il peut être facilement connecté à un classifieur (ex: LogisticRegression) pour faire de la prédiction de prdtypecode.

Pré-requis :
- spaCy installé avec le modèle français : `python -m spacy download fr_core_news_sm`
- Fichiers CSV : X_train_update.csv et Y_train_CVw08PX.csv dans le dossier `../data`
"""

import pandas as pd
import re
import unicodedata
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

X_train = pd.read_csv('../data/X_train_update.csv')
y_train = pd.read_csv('../data/Y_train_CVw08PX.csv')

df = X_train.copy()
df['prdtypecode'] = y_train['prdtypecode']
nlp = spacy.load("fr_core_news_sm")

STOPWORD = set([
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux", "en",
    "et", "à", "pour", "par", "avec", "sur", "dans", "ce", "ces", "se", "sa",
    "son", "ses", "qui", "que", "quoi", "dont", "où", "comme", "est", "sont",
    "il", "elle", "ils", "elles", "nous", "vous", "ne", "pas", "plus", "moins",
    "ou", "mais", "donc", "or", "ni", "car"
])

def remove_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.lemma_ not in STOPWORD and len(token.lemma_) > 2 and not token.is_punct and not token.is_space])

LANGUES = {
    "en": "Helsinki-NLP/opus-mt-en-fr",
    "it": "Helsinki-NLP/opus-mt-it-fr",
    "es": "Helsinki-NLP/opus-mt-es-fr",
    "nl": "Helsinki-NLP/opus-mt-nl-fr",
    "ro": "Helsinki-NLP/opus-mt-ro-fr"
}

MODELE_TRADUCTION = {}

def detection_langue(text):
    try:
        return detect(text)
    except LangDetectException:
        return "inconnu"
    
def traduction(text, lang_code):
    if lang_code not in LANGUES:
        return text
    if lang_code not in MODELE_TRADUCTION:
        model_name = LANGUES[lang_code]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        MODELE_TRADUCTION[lang_code] = (tokenizer, model)
    else:
        tokenizer, model = MODELE_TRADUCTION[lang_code]

    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

class NettoyeurTexte(BaseEstimator, TransformerMixin):
    """
    Nettoyeur de texte personnalisé pour scikit-learn.

    Applique un nettoyage linguistique de base :
    - Passage en minuscules
    - Suppression des accents
    - Suppression des balises HTML, ponctuation et chiffres
    - Lemmatisation avec spaCy (fr_core_news_sm)
    - Suppression des stopwords et des tokens courts
    """

    def fit(self, X: pd.DataFrame, y=None) -> "NettoyeurTexte":
        """
        Ne fait rien. Nécessaire pour compatibilité avec l'API scikit-learn.

        Args:
            X (pd.DataFrame): DataFrame avec au moins les colonnes 'description' et 'designation'.
            y (Any, optional): Données cibles (non utilisées ici). Defaults to None.

        Returns:
            NettoyeurTexte: L'instance de ce transformateur (self).
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> list[str]:
        """
            Applique le nettoyage et la lemmatisation à chaque ligne du DataFrame.

            Args:
                X (pd.DataFrame): DataFrame contenant les colonnes 'description' et 'designation'.
                y (Any, optional): Ignoré.

            Returns:
                list[str]: Liste de chaînes nettoyées et lemmatisées.
        """
        texts = []
        for _, row in X.iterrows():
            text = row['description']
            if pd.isnull(text) or str(text).strip() == "":
                text = row['designation']
            if pd.isnull(text):
                texts.append("")
                continue

            text = text.lower()
            text = remove_accents(text)

            lang = detection_langue(text)
            if lang in LANGUES:
                text = traduction(text, lang)

            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\b\w*\d\w*\b', ' ', text)
            text = re.sub(r'[^a-z\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            texts.append(text)

        docs = nlp.pipe(texts, disable=["parser", "ner"])
        cleaned = []
        for doc in docs:
            lemmas = [
                token.lemma_
                for token in doc
                if token.lemma_ not in STOPWORD
                and len(token.lemma_) > 2
                and not token.is_punct
                and not token.is_space
            ]
            cleaned.append(" ".join(lemmas))

        return cleaned

df_sample = df[['description', 'designation']].sample(5, random_state=42)
nettoyeur = NettoyeurTexte()
df_sample['description_propre'] = nettoyeur.transform(df_sample)
print(df_sample['description_propre'])

df_cleaned = df.copy()
df_cleaned['description'] = nettoyeur.transform(df_cleaned[['description', 'designation']])

df_cleaned.to_csv("X_train_cleaned2.csv", index=False)
print(" Données sauvegardées dans X_train_cleaned.csv avec les descriptions nettoyées.")

"""
Pipeline de traitement texte : 

    Nettoyage des colonnes 'description' et 'designation' avec le transformateur personnalisé NettoyeurTexte.
    → Convertit les textes en minuscules, enlève accents, ponctuation, stopwords, et applique la lemmatisation.

    Vectorisation TF-IDF avec TfidfVectorizer :
    → Utilise unigrams uniquement, limite à 1000 mots les plus fréquents (max_features)
    → Applique un filtrage simple des stopwords anglais    
"""

pipeline = Pipeline([
    ('nettoyage', NettoyeurTexte()),
    ('tfidf', TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 1),  
        stop_words='english'  
    ))
    
])

X_tfidf = pipeline.fit_transform(df[['description', 'designation']])

print("TF-IDF shape :", X_tfidf.shape)

"""
Ce bloc extrait et affiche un exemple de la matrice TF-IDF après transformation.

Étapes :
    Récupère le TfidfVectorizer utilisé dans le pipeline via named_steps.
    Extrait les noms des features (mots/ngrammes) générés par le vectoriseur.
    Convertit les 5 premières lignes de la matrice sparse TF-IDF en tableau dense.
    Construit un DataFrame pandas lisible avec les colonnes nommées selon les mots.
    Supprime les colonnes où tous les scores TF-IDF sont nuls pour simplifier l'affichage.
    Affiche les lignes non nulles pour visualiser concrètement l'encodage texte → vecteurs.

Utile pour débug ou comprendre la représentation vectorielle des textes.
"""
tfidf_vectorizer = pipeline.named_steps['tfidf']
feature_names = tfidf_vectorizer.get_feature_names_out()
sample_tfidf = X_tfidf[:5].toarray()
df_tfidf_sample = pd.DataFrame(sample_tfidf, columns=feature_names)
df_tfidf_sample = df_tfidf_sample.loc[:, (df_tfidf_sample != 0).any(axis=0)]
print("Exemple fictif de matrice TF-IDF :")
print(df_tfidf_sample.head())

feature_names = pipeline.get_feature_names_out()
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=feature_names)
df_tfidf.insert(0, "productid", df['productid'].values)
df_tfidf.to_csv("tfidf_vectorized_matrix.csv", index=False)
print("Matrice TF-IDF enregistrée dans : tfidf_vectorized_matrix.csv")
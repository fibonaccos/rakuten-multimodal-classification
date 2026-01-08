import pandas as pd
import re
import unicodedata
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

# Configuration
STOPWORDS = set(["le", "la", "les", "un", "une", "des", "de", "du", "et", "à", "pour", "que", "qui", "dans", "en", "sur"]) # Liste abrégée pour l'exemple
try:
    nlp = spacy.load("fr_core_news_sm")
except:
    print("Modèle spaCy 'fr_core_news_sm' introuvable.")
    nlp = None

def remove_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

class NettoyeurTexte(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            data = X.copy()
        else:
            data = pd.DataFrame({'description': X, 'designation': [''] * len(X)})

        texts = []
        for row in data.itertuples(index=False):
            desc = getattr(row, 'description', '') if pd.notnull(getattr(row, 'description', '')) else ''
            desig = getattr(row, 'designation', '') if pd.notnull(getattr(row, 'designation', '')) else ''
            
            text = desc if str(desc).strip() else desig
            
            text = str(text).lower()
            text = remove_accents(text)
            text = re.sub(r'<[^>]+>', ' ', text) 
            text = re.sub(r'[^a-z\s]', ' ', text) 
            text = re.sub(r'\s+', ' ', text).strip()
            texts.append(text)

        if nlp:
            docs = nlp.pipe(texts, disable=["parser", "ner"])
            cleaned = []
            for doc in docs:
                lemmas = [t.lemma_ for t in doc if t.lemma_ not in STOPWORDS and len(t.lemma_) > 2]
                cleaned.append(" ".join(lemmas))
            return cleaned
        return texts

def preprocess_data(input_path, output_path):
    print(f"[Preprocessing] Chargement : {input_path}")
    df = pd.read_csv(input_path)
    
    cleaner = NettoyeurTexte()
    cols = df[['description', 'designation']] if 'designation' in df.columns else df[['description']]
    
    df['description_propre'] = cleaner.transform(cols)
    
    cols_to_save = ['description_propre']
    if 'prdtypecode' in df.columns: cols_to_save.append('prdtypecode')
    if 'productid' in df.columns: cols_to_save.append('productid')
    
    df[cols_to_save].to_csv(output_path, index=False)
    print(f"[Preprocessing] Sauvegardé : {output_path}")
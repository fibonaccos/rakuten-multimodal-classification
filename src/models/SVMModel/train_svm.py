import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from src.preprocessing.SVMModel.preprocessing import NettoyeurTexte
from src.models.SVMModel.shap_interpret import explain_pipeline_shap

def train_model(clean_data_path, model_out_path, artifacts_dir):
    print(f"[Train] Chargement données : {clean_data_path}")
    df = pd.read_csv(clean_data_path)
    
    df['description_propre'] = df['description_propre'].fillna("").astype(str)
    
    X = df['description_propre']
    y = df['prdtypecode']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LinearSVC(random_state=42))
    ])
    
    model_pipe.fit(X_train, y_train)
    print(f"[Train] Score Accuracy : {model_pipe.score(X_test, y_test):.4f}")
    
    full_pipeline = Pipeline([
        ('preprocessor', NettoyeurTexte()), 
        ('tfidf', model_pipe.named_steps['tfidf']),
        ('clf', model_pipe.named_steps['clf'])
    ])
    
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    joblib.dump(full_pipeline, model_out_path)
    print(f"[Train] Modèle sauvegardé : {model_out_path}")

    explain_pipeline_shap(model_pipe, X_train, X_test, artifacts_dir)
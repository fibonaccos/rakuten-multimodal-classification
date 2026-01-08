import joblib
import pandas as pd
from src.preprocessing.SVMModel.preprocessing import NettoyeurTexte

def predict_new(model_path, texts):
    """
    Predit la classe pour une liste de textes bruts.
    """
    print(f"Chargement du modèle depuis {model_path}...")
    pipeline = joblib.load(model_path)
    
    df_input = pd.DataFrame({'description': texts, 'designation': [''] * len(texts)})
    
    print("Prédiction en cours...")
    predictions = pipeline.predict(df_input)
    return predictions

if __name__ == "__main__":
    mod_path = "models/svm_rakuten.joblib"
    exemples = ["Piscine gonflable pour enfants", "Jeu vidéo PS5 Spiderman"]
    try:
        preds = predict_new(mod_path, exemples)
        for t, p in zip(exemples, preds):
            print(f"Texte: {t} -> Prédiction: {p}")
    except FileNotFoundError:
        print("Modèle introuvable. Lancez l'entraînement d'abord.")
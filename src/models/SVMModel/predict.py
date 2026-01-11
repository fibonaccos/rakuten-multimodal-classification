import joblib
import pandas as pd
import os
import re
import unicodedata

# --- Fonction de nettoyage interne pour les nouvelles prédictions ---
def clean_text_for_prediction(text):
    if not isinstance(text, str): return ""
    # 1. Décoder et minuscule
    text = text.lower()
    # 2. Retirer accents
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    # 3. Regex (garder lettres et espaces, retirer chiffres isolés si nécessaire)
    text = re.sub(r'<[^>]+>', ' ', text)       # HTML
    text = re.sub(r'[^a-z\s]', ' ', text)      # Lettres a-z
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_svm(input_data, model_path):
    print(f"Chargement du modèle depuis : {model_path}")
    
    if not os.path.exists(model_path):
        print("Erreur : Modèle introuvable. Veuillez lancer l'entraînement d'abord.")
        return

    # Chargement du pipeline complet
    pipeline = joblib.load(model_path)
    
    texts_to_predict = []
    
    # Cas 1 : input_data est un fichier CSV
    if input_data.endswith('.csv') and os.path.exists(input_data):
        print(f"Prédiction sur le fichier : {input_data}")
        df = pd.read_csv(input_data)
        
        # On cherche la colonne pertinente
        col = 'description' if 'description' in df.columns else 'designation'
        if col not in df.columns:
            print("Erreur : Colonne 'description' ou 'designation' manquante.")
            return
            
        # On applique le nettoyage AVANT de prédire
        texts_to_predict = df[col].fillna("").astype(str).apply(clean_text_for_prediction)
        
    # Cas 2 : input_data est une phrase brute
    else:
        print("Prédiction sur le texte brut.")
        # On nettoie la phrase
        cleaned_text = clean_text_for_prediction(input_data)
        texts_to_predict = [cleaned_text]
        print(f"Texte nettoyé : {cleaned_text}")

    # Prédiction
    predictions = pipeline.predict(texts_to_predict)

    # Affichage
    if len(predictions) == 1:
        print(f"\n>>> Code Produit Prédit : {predictions[0]} <<<\n")
    else:
        print(f"{len(predictions)} prédictions effectuées.")
        # Sauvegarde optionnelle
        pd.DataFrame({'text': texts_to_predict, 'prediction': predictions}).to_csv("resultats_prediction.csv", index=False)
        print("Résultats sauvegardés dans 'resultats_prediction.csv'")
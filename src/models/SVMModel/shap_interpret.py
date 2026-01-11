import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def interpret_shap(model_path, data_path, num_samples=100):
    print("Chargement du modèle et des données pour SHAP...")
    
    if not os.path.exists(model_path):
        print(f"Erreur : Modèle introuvable à l'adresse {model_path}")
        return

    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return

    df = pd.read_csv(data_path)
    
    col_text = 'description' if 'description' in df.columns else 'designation'
    X_sample = df[col_text].fillna("").sample(n=num_samples, random_state=42)
    
    try:
        vectorizer = pipeline.named_steps['tfidf']
        classifier = pipeline.named_steps['clf']
    except KeyError:
        print("Erreur : Le pipeline ne contient pas les étapes nommées 'tfidf' et 'clf'.")
        print("Étapes trouvées :", pipeline.named_steps.keys())
        return
    
    X_sample_vec = vectorizer.transform(X_sample)
    
    print("Calcul des valeurs SHAP (cela peut prendre quelques secondes)...")
    explainer = shap.LinearExplainer(classifier, X_sample_vec, feature_perturbation="interventional")
    
    shap_values = explainer.shap_values(X_sample_vec)

    print("Génération du graphique...")
    feature_names = vectorizer.get_feature_names_out()
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample_vec, feature_names=feature_names, show=False)
    plt.title("Mots les plus impactants (SHAP)")
    plt.tight_layout()
    
    output_img = "shap_summary.png"
    plt.savefig(output_img)
    print(f"Graphique sauvegardé sous : {output_img}")
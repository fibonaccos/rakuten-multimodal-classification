import pandas as pd
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

def train_svm(data_path, model_path, report_dir="reports/SVMModel"):
    print(f"Chargement des données depuis : {data_path}")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {data_path} est introuvable.")
        return

    if 'description' not in df.columns or 'prdtypecode' not in df.columns:
        print("Erreur : Les colonnes 'description' et 'prdtypecode' sont requises.")
        return

    X = df['description'].fillna("").astype(str)
    y = df['prdtypecode']

    print("Split des données (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Entraînement du modèle (TF-IDF + SGD)...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3)),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, random_state=42, max_iter=1000, tol=1e-3, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)

    print(f"Génération des rapports dans : {report_dir} ...")
    os.makedirs(report_dir, exist_ok=True)
    
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted')
    }
    with open(os.path.join(report_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

        report_str = classification_report(y_test, y_pred)
    with open(os.path.join(report_dir, "classification_report.txt"), "w") as f:
        f.write(report_str)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(os.path.join(report_dir, "classification_report.csv"))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 12))
    sorted_labels = sorted(y.unique())
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=sorted_labels, yticklabels=sorted_labels)
    plt.title("Matrice de Confusion SVM")
    plt.ylabel('Vrai Label')
    plt.xlabel('Label Prédit')
    plt.savefig(os.path.join(report_dir, "confusion_matrix.png"))
    plt.close()

    print(f"Rapports sauvegardés (dont classification_report.csv).")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Modèle sauvegardé sous : {model_path}")
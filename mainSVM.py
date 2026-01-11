import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.SVMModel.train_svm import train_svm
from src.models.SVMModel.predict import predict_svm
from src.models.SVMModel.shap_interpret import interpret_shap

# --- CONFIGURATION ---
DATA_CLEANED = "data/processed/X_train_cleaned.csv" 
MODEL_PATH = "models/SVMModel/svm_pipeline.pkl"
REPORT_DIR = "reports/SVMModel"

def main():
    parser = argparse.ArgumentParser(description="Rakuten Classification Pipeline (SVM)")
    parser.add_argument("--step", type=str, required=True, 
                        choices=["train", "predict", "interpret"], 
                        help="Action à effectuer")
    parser.add_argument("--text", type=str, help="Texte à prédire")
    parser.add_argument("--file", type=str, help="Fichier CSV pour prédiction")

    args = parser.parse_args()

    # Vérification fichier train
    if args.step in ["train", "interpret"] and not os.path.exists(DATA_CLEANED):
        print(f"ERREUR : Fichier introuvable : {DATA_CLEANED}")
        return

    if args.step == "train":
        # On passe REPORT_DIR à la fonction train_svm
        train_svm(DATA_CLEANED, MODEL_PATH, report_dir=REPORT_DIR)

    elif args.step == "predict":
        if args.text:
            predict_svm(args.text, MODEL_PATH)
        elif args.file:
            predict_svm(args.file, MODEL_PATH)
        else:
            print("Test prédiction défaut :")
            predict_svm("piscine gonflable enfant jardin", MODEL_PATH)

    elif args.step == "interpret":
        interpret_shap(MODEL_PATH, DATA_CLEANED)

if __name__ == "__main__":
    main()
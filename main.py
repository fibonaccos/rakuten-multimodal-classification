import argparse
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.SVMModel.preprocessing import preprocess_data
from src.models.SVMModel.train_svm import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["preprocess", "train", "all"], default="all")
    args = parser.parse_args()

    RAW_X = "data/raw/X_train_update.csv"
    RAW_Y = "data/raw/Y_train_CVw08PX.csv"
    TEMP_MERGED = "data/raw/merged_temp.csv"
    CLEAN_DATA = "data/processed/X_train_cleaned.csv"
    MODEL_PATH = "models/SVM/svm_model.joblib"
    OUTPUT_DIR = "reports/SVM"

    if args.step in ["preprocess", "all"]:
        if os.path.exists(RAW_X) and os.path.exists(RAW_Y):
            dx = pd.read_csv(RAW_X)
            dy = pd.read_csv(RAW_Y)
            dx['prdtypecode'] = dy['prdtypecode']
            dx.to_csv(TEMP_MERGED, index=False)
            
            preprocess_data(TEMP_MERGED, CLEAN_DATA)
            
            os.remove(TEMP_MERGED)
        else:
            print(f"Erreur: Fichiers {RAW_X} ou {RAW_Y} manquants.")

    if args.step in ["train", "all"]:
        if os.path.exists(CLEAN_DATA):
            train_model(CLEAN_DATA, MODEL_PATH, OUTPUT_DIR)
        else:
            print("Erreur: Lancez l'étape preprocess d'abord.")

if __name__ == "__main__":
    main()
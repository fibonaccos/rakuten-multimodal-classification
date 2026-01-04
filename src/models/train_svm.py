
import argparse
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from shap_interpret import explain_pipeline_shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Train a Linear SVM on Rakuten cleaned text (verbose).")
    p.add_argument("--csv", type=str, default="X_train_cleaned2.csv",
                   help="Path to cleaned CSV with columns: description, prdtypecode, (optional) productid.")
    p.add_argument("--test_size", type=float, default=0.2, help="Test split size.")
    p.add_argument("--seed", type=int, default=42, help="Random state.")
    p.add_argument("--max_features", type=int, default=5000, help="TF-IDF max features.")
    p.add_argument("--bigrams", action="store_true", help="Use bigrams (ngram_range=(1,2)).")
    p.add_argument("--outdir", type=str, default="outputs", help="Folder to save reports and figures.")
    p.add_argument("--save_preds", type=str, default="", help="CSV path to save detailed predictions.")
    p.add_argument("--save_model", type=str, default="", help="Path to save trained model with joblib.")
    p.add_argument("--sample", type=int, default=0, help="If >0, randomly subsample this many rows for a quick run.")
    return p.parse_args()


def load_data(path, sample=0, seed=42):
    print(f"[1/6] téléchargement du CSV: {path}")
    df = pd.read_csv(path)
    print(f" shape: {df.shape}")
    assert "description" in df.columns, "no 'description' dans le CSV."
    assert "prdtypecode" in df.columns, "no 'prdtypecode' dans le CSV."
    df["description"] = df["description"].fillna("").astype(str)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
        print(f"subsampled: {df.shape}")
    X = df["description"].values
    y = df["prdtypecode"].values
    pid = df["productid"].values if "productid" in df.columns else np.arange(len(df))
    return X, y, pid


def build_pipeline(max_features=5000, bigrams=False, class_weight="balanced"):
    ngram = (1, 2) if bigrams else (1, 1)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram,
            norm="l2",
            lowercase=False
        )),
        ("clf", LinearSVC(
            C=1.0,
            class_weight=class_weight,
            random_state=42
        ))
    ])
    return pipe


def save_eval_artifacts(y_test, y_pred, labels, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)

    # classification report dict -> csv (déjà présent)
    report_dict = classification_report(y_test, y_pred, digits=4, output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(os.path.join(outdir, "classification_report.csv"))

    # On prend uniquement les lignes correspondant aux classes (precision, recall, f1-score) si présent
    stats = ["precision", "recall", "f1-score", "support"]

    cr_rows = []
    for lab in labels:
        if lab in report_df.index:
            cr_rows.append(report_df.loc[lab, stats].values)
        else:
            cr_rows.append([0.0, 0.0, 0.0, 0.0])
    cr_mat = np.array(cr_rows, dtype=float)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.15 + 6), 8))
    im = ax.imshow(cr_mat[:, :3], aspect='auto', interpolation='nearest') 
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(stats[:3])

    for i in range(cr_mat.shape[0]):
        for j in range(3):
            text = f"{cr_mat[i, j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="w" if cr_mat[i, j] < 0.5 else "black")
    ax.set_title("Classification report (precision / recall / f1) par class")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    png_path = os.path.join(outdir, "classification.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Sauvegarde: {png_path}")

    # Enregistrement de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in labels], columns=[f"pred_{c}" for c in labels])
    cm_df.to_csv(os.path.join(outdir, "Matrice_de_confusion.csv"))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Matrice de confusion")
    ax.set_xlabel("Label prédit")
    ax.set_ylabel("Bon label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, "matrice_confusion.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    try:
        X, y, pid = load_data(args.csv, sample=args.sample, seed=args.seed)
    except FileNotFoundError as e:
        print(f"Fichier non trouvé: {e}")
        return
    except AssertionError as e:
        print(f"CSV schema erreur: {e}")
        return
    except Exception as e:
        print(f"Fail téléchargement des données: {e}")
        return

    print(f"[2/6] Split train/test (test_size={args.test_size})")
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    print(f"[3/6] Pipeline (max_features={args.max_features}, bigrams={args.bigrams})")
    pipe = build_pipeline(max_features=args.max_features, bigrams=args.bigrams, class_weight="balanced")

    print("[4/6] modélisation...")
    pipe.fit(X_train, y_train)

    print("[5/6] Prédiction...")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    print("\n Résultats")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1-macro  : {f1m:.4f}")
    print("\n Classification report")
    print(classification_report(y_test, y_pred, digits=4))
    print("\n Matrice de confusion")
    print(confusion_matrix(y_test, y_pred))

    print("[6/6] Enregistrement...")
    labels = pipe.named_steps['clf'].classes_
    save_eval_artifacts(y_test, y_pred, labels, outdir=args.outdir)

    try:
        
        print("[7/7] SHAP tourne")
        
        explain_pipeline_shap(
            pipeline=pipe,
            X_train_raw=X_train,       
            X_explain_raw=X_test[:100], 
            outdir=args.outdir,
            background_sample=200,      
            top_features=30
        )
        print("SHAP artifacts saved in:", args.outdir)
    except Exception as e:
        print("SHAP interpretability skipped or failed:", e)

    if args.save_preds:
        scores = pipe.decision_function(X_test)
        margins = scores.max(axis=1)
        test_pids = pid[idx_test]
        preds_df = pd.DataFrame({
            "productid": test_pids,
            "true_label": y_test,
            "pred_label": y_pred,
            "correct": (y_test == y_pred),
            "margin": margins
        })
        preds_df.to_csv(args.save_preds, index=False)
        print(f"Sauvegarde: {args.save_preds}")

    if args.save_model:
        try:
            import joblib
            joblib.dump(pipe, args.save_model)
            print(f"Sauvegarde: {args.save_model}")
        except Exception as e:
            print(f"Fail: {e}")


if __name__ == "__main__":
    main()

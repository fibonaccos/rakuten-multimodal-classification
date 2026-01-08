import shap
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import os

def explain_pipeline_shap(pipeline, X_train_cleaned, X_test_cleaned, outdir):
    os.makedirs(outdir, exist_ok=True)
    
    vect = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    
    X_bg = vect.transform(X_train_cleaned[:100]).toarray()
    X_exp = vect.transform(X_test_cleaned[:20]).toarray()
    feature_names = vect.get_feature_names_out()

    print("[SHAP] Calcul des valeurs SHAP...")
    masker = shap.maskers.Independent(X_bg)
    explainer = shap.LinearExplainer(clf, masker=masker)
    shap_values = explainer.shap_values(X_exp)

    plt.figure()
    shap.summary_plot(shap_values, X_exp, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(outdir, "shap_summary.png"), bbox_inches='tight')
    plt.close()
    print(f"[SHAP] Graphique sauvegardé dans {outdir}")
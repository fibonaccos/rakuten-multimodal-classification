import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

def explain_pipeline_shap(
    pipeline,
    X_train_brut,
    X_a_expliquer,
    repertoire_sortie="outputs",
    taille_fond=200,
    top_caracteres=30,
    etat_aleatoire=42
):
    os.makedirs(repertoire_sortie, exist_ok=True)

    # Récupérer le vectoriseur et le classifieur (heuristique)
    if hasattr(pipeline, "named_steps") and "tfidf" in pipeline.named_steps and "clf" in pipeline.named_steps:
        vectoriseur = pipeline.named_steps["tfidf"]
        classifieur = pipeline.named_steps["clf"]
    else:
        # fallback : première étape = vectoriseur, dernière étape = classifieur
        vectoriseur = pipeline.steps[0][1]
        classifieur = pipeline.steps[-1][1]

    X_train_brut = np.array(X_train_brut)
    X_a_expliquer = np.array(X_a_expliquer)
    rng = np.random.RandomState(etat_aleatoire)
    n_fond = min(taille_fond, len(X_train_brut))
    if n_fond <= 0:
        raise ValueError("taille_fond doit être > 0")
    idx_fond = rng.choice(len(X_train_brut), size=n_fond, replace=False)
    X_fond_texte = X_train_brut[idx_fond]

    # Vectorisation
    X_fond_vec = vectoriseur.transform(X_fond_texte)
    X_explain_vec = vectoriseur.transform(X_a_expliquer)

    X_fond_arr = X_fond_vec.toarray()
    X_explain_arr = X_explain_vec.toarray()

    # Récupérer les noms de features
    try:
        noms_features = vectoriseur.get_feature_names_out()
    except Exception:
        noms_features = [f"f{i}" for i in range(X_fond_arr.shape[1])]


    explainer = None
    try:
        # Essayer la construction avec masque (masker)
        try:
            masker = None
            try:
                masker = shap.maskers.Independent(X_fond_arr)
            except Exception:
                masker = X_fond_arr
            try:
                explainer = shap.LinearExplainer(classifieur, masker=masker)
            except Exception:
                try:
                    explainer = shap.LinearExplainer(classifieur, X_fond_arr, masker=masker)
                except Exception:
                    explainer = None
        except Exception:
            explainer = None

        # Essayer différentes options de feature_perturbation si nécessaire
        if explainer is None:
            for fp in ("interventional", "correlation_dependent"):
                try:
                    explainer = shap.LinearExplainer(classifieur, X_fond_arr, feature_perturbation=fp)
                    break
                except Exception:
                    explainer = None

        # Tentative simple
        if explainer is None:
            try:
                explainer = shap.LinearExplainer(classifieur, X_fond_arr)
            except Exception:
                explainer = None

        # Fallback final 
        if explainer is None:
            try:
                try:
                    masker = shap.maskers.Independent(X_fond_arr)
                except Exception:
                    masker = X_fond_arr
                explainer = shap.Explainer(classifieur, masker)
            except Exception as e:
                raise RuntimeError("Impossible de construire un explainer SHAP avec cette version de shap.") from e
    except Exception as e:
        raise RuntimeError("Erreur lors de la création de l'explainer SHAP: " + str(e)) from e

    
    raw_shap = explainer.shap_values(X_explain_arr)

   
    if hasattr(raw_shap, "values"):
        vals = raw_shap.values
    else:
        vals = raw_shap

    multiclass = False
    single_array = False

    if isinstance(vals, list):
        vals = [np.asarray(v) for v in vals]
        shapes = [v.shape for v in vals]
        # liste d'arrays
        if all(len(s) == 2 for s in shapes) and len({s[1] for s in shapes}) == 1:
            multiclass = True
            n_classes = len(vals)
            n_examples, n_features = vals[0].shape
            per_class_mean_abs = np.array([np.mean(np.abs(v), axis=0) for v in vals])
            abs_mean = np.mean(per_class_mean_abs, axis=0)
        else:
            # essayer stacked 
            try:
                stacked = np.stack(vals, axis=0)
                if stacked.ndim == 3:
                    multiclass = True
                    n_classes, n_examples, n_features = stacked.shape
                    vals = [stacked[c] for c in range(n_classes)]
                    abs_mean = np.mean(np.array([np.mean(np.abs(v), axis=0) for v in vals]), axis=0)
                else:
                    raise ValueError("unexpected stacked ndim")
            except Exception:
                # fallback : calculer la moyenne absolue par tableau quand possible
                all_abs_means = []
                for v in vals:
                    v = np.asarray(v)
                    if v.ndim == 2:
                        all_abs_means.append(np.mean(np.abs(v), axis=0))
                if len(all_abs_means) == 0:
                    raise RuntimeError(f"Impossible d'interpréter la structure retournée par shap (list) ; shapes={shapes}")
                abs_mean = np.mean(np.stack(all_abs_means, axis=0), axis=0)
                multiclass = True
                n_classes = len(all_abs_means)
                n_features = abs_mean.shape[0]

    elif isinstance(vals, np.ndarray):
        vals = np.asarray(vals)
        if vals.ndim == 3:
            multiclass = True
            n_classes, n_examples, n_features = vals.shape
            vals = [vals[c] for c in range(n_classes)]
            abs_mean = np.mean(np.array([np.mean(np.abs(v), axis=0) for v in vals]), axis=0)
        elif vals.ndim == 2:
            single_array = True
            n_examples, n_features = vals.shape
            abs_mean = np.mean(np.abs(vals), axis=0)
            
        else:
            raise RuntimeError(f"Format inattendu pour shap_values ndarray ndim={vals.ndim}, shape={vals.shape}")
    else:
        raise RuntimeError(f"Format inattendu pour shap_values : {type(vals)}")

    
    top_idx = np.argsort(abs_mean)[-top_caracteres:][::-1]
    top_feat_names = [noms_features[i] for i in top_idx]
    top_vals = abs_mean[top_idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(top_feat_names) * 0.25 + 2)))
    y_pos = np.arange(len(top_feat_names))
    ax.barh(y_pos[::-1], top_vals[::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feat_names[::-1])
    ax.set_xlabel("Moyenne(|valeurs SHAP|)")
    ax.set_title("Top des caractéristiques par valeur SHAP moyenne (globale)")
    fig.tight_layout()
    out_path = os.path.join(repertoire_sortie, "shap_summary_top_features.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Enregistré dans : {out_path}")

    # Récupérer le vecteur SHAP pour la classe prédite 
    def get_shap_for_sample(i, class_idx=None):
        if single_array:
            return vals[i]
        else:
            if class_idx is None:
                try:
                    pred = classifieur.predict(X_explain_arr[i:i+1])[0]
                    class_idx = int(np.where(classifieur.classes_ == pred)[0][0])
                except Exception:
                    class_idx = 0
            return vals[class_idx][i]

    # Graphiques SHAP par échantillon
    n_explain = min(len(X_a_expliquer), 50)  
    for i in range(n_explain):
        try:
            sv = get_shap_for_sample(i)
        except Exception as e:
            print(f"Skipping sample {i} due to error getting SHAP vector: {e}")
            continue

        # sélectionner les features importantes pour cet échantillon 
        sel_idx = np.argsort(np.abs(sv))[-top_caracteres:][::-1]
        sel_names = [noms_features[j] for j in sel_idx]
        sel_vals = sv[sel_idx]

        fig, ax = plt.subplots(figsize=(8, max(3, len(sel_names) * 0.25 + 1)))
        colors = ['green' if v > 0 else 'red' for v in sel_vals]
        y_pos = np.arange(len(sel_names))
        ax.barh(y_pos[::-1], sel_vals[::-1], color=colors[::-1])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sel_names[::-1])
        try:
            pred = classifieur.predict(X_explain_arr[i:i+1])[0]
        except Exception:
            pred = None
        ax.set_xlabel("Valeur SHAP")
        ax.set_title(f"Échantillon {i} - top {len(sel_names)} features (pred={pred})")
        fig.tight_layout()
        sample_path = os.path.join(repertoire_sortie, f"shap_sample_{i}.png")
        plt.savefig(sample_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"Enregistré les graphiques SHAP par échantillon (jusqu'à {n_explain} échantillons) dans : {repertoire_sortie}")
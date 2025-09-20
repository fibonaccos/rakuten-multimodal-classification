Project Name
==============================

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

# README pour le workflow SGD robuste

# SGD_Training_Colab_Robust_GridSearch

Ce dépôt contient le workflow le plus robuste pour l'entraînement d'un modèle SGD multi-classes sur les données Rakuten, optimisé pour Google Colab et les interruptions longues.

## Fichiers nécessaires
- `SGD_Training_Colab_Robust_GridSearch.ipynb` : Notebook principal à lancer sur Colab.
- `data/processed/features_for_dt.csv` : Fichier de features à utiliser pour l'entraînement.
- Dossiers de sortie (créés automatiquement) : `models/`, `reports/`, `reports/figures/`, `checkpoints/`

## Utilisation sur Google Colab
1. **Uploader le dossier sur Google Drive**
2. **Ouvrir Google Colab**
3. **Monter le Drive** :
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. **Adapter le chemin du projet** dans le notebook :
   ```python
   PROJECT_PATH = '/content/drive/MyDrive/rakuten-multimodal-classification'
   os.chdir(PROJECT_PATH)
   ```
5. **Lancer la cellule principale** du notebook. Le GridSearch se fait par lots, avec checkpoints automatiques. Si le runtime se déconnecte, relance simplement la cellule : le script reprendra là où il s'est arrêté.
6. **À la fin**, une archive contenant les modèles, rapports et figures sera générée et téléchargeable.

## Nettoyage du dépôt
Supprime tous les fichiers/dossiers suivants pour ne garder que le workflow robuste :
- `train_sgd.py`, `train_sgd_fixed.py`, `monitor_training.py`, `diagnose_sgd_issues.py`, `optimize_sgd.py`, `check_data.py`, `sgd_colab_script.py`, `prepare_colab.py`
- `config.json`, `config_sgd_improved.json`
- `notebooks/`, `rakuten_sgd_colab_250915_132807/`, `rakuten_sgd_colab_250919_114644/`, `sgd_results_250916-103142/`
- `GUIDE_SGD_IMPLEMENTATION.md`, `SGD_IMPLEMENTATION_GUIDE.md`, `README_COLAB.md`, `preprocessing_config_helper.md`, `to_do.md`
- Tous les rapports/figures non générés par le script robuste

## Conseils
- Conserve une sauvegarde temporaire avant suppression définitive.
- Ajoute un `.gitignore` pour ignorer les dossiers de logs, checkpoints, modèles, etc.

## Auteur
Mise à jour : 20/09/2025

---
Pour toute question, consulte le notebook principal ou contacte le mainteneur du dépôt.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

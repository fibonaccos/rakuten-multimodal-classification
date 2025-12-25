# Rakuten Multimodal Product Classification

Classification multimodale (texte + image) de produits e-commerce Rakuten France.  
Challenge Data ENS : https://challengedata.ens.fr/challenges/35

## 📋 Description du Projet

Ce projet vise à classifier automatiquement des produits e-commerce dans leurs catégories respectives en utilisant à la fois les données textuelles (désignation et description) et les images des produits. Il s'inscrit dans le cadre du **Rakuten France Multimodal Product Data Classification Challenge**.

### Objectif
Prédire le code type de 27 catégories de produits à partir de :
- **Données textuelles** : titres et descriptions (~60 MB)
- **Données images** : images des produits (~2.2 GB)
- **Métrique** : Weighted F1-Score

### Contexte
99k produits répartis dans plus de 1000 classes avec une distribution déséquilibrée. Le défi présente des aspects de recherche intéressants dus à la nature intrinsèquement bruitée des étiquettes et images de produits.

## 🎯 Modèles Implémentés

Le projet explore plusieurs approches de classification :

### 1. **Modèles Textuels**
- **Decision Tree** : Classification basée sur les features textuelles
- **SGD Classifier** : Stochastic Gradient Descent avec GridSearch pour l'optimisation
- **SVM** : Support Vector Machine pour la classification multiclasse

### 2. **Modèles Images**
- **CNN Custom** : Réseaux de neurones convolutifs personnalisés
- **ResNet** : Transfer learning avec ResNet pre-entraîné
- **YOLO** : Détection et classification d'objets

### 3. **Modèles Multimodaux**
- Combinaison des features textuelles et visuelles
- Fusion de prédictions de différents modèles

### 4. **Interprétabilité**
- **SHAP** : Explications des prédictions des modèles


## 📁 Structure du Projet

    ├── LICENSE
    ├── README.md                   <- Documentation principale du projet
    ├── requirements.txt            <- Dépendances Python du projet
    ├── config.json                 <- Configuration du projet
    │
    ├── data                        <- Données (non versionnées sur Git)
    │   ├── raw                     <- Données brutes originales
    │   ├── processed               <- Données traitées pour la modélisation
    │   └── images                  <- Images des produits
    │
    ├── models                      <- Modèles entraînés et sérialisés
    │   └── cnn                     <- Configuration et modèles CNN
    │       └── cnn_model_config_helper.md
    │
    ├── notebooks                   <- Jupyter notebooks organisés par thème
    │   ├── exploration             <- Exploration et analyse des données
    │   │   ├── exploration-image-1.ipynb
    │   │   ├── exploration-text-1.ipynb
    │   │   ├── exploration-text-2.ipynb
    │   │   └── rakuten_exploration_text.ipynb
    │   ├── models                  <- Notebooks d'entraînement des modèles
    │   │   ├── SGD_Training_Colab_Robust_GridSearch.ipynb
    │   │   ├── rakuten_resnet.ipynb
    │   │   └── rakuten_yolo.ipynb
    │   └── preprocessing           <- Notebooks de prétraitement
    │       ├── nettoyage.ipynb
    │       └── rakuten_preprocessing_image.ipynb
    │
    ├── reports                     <- Rapports et documentation
    │   ├── figures                 <- Graphiques et visualisations
    │   ├── models                  <- Documentation des modèles
    │   │   └── SGD_Colab_Workflow.md
    │   ├── methodologie_rapport.md
    │   └── preprocessing_config_helper.md
    │
    ├── references                  <- Dictionnaires de données, manuels
    │
    ├── logs                        <- Logs d'exécution
    │
    └── src                         <- Code source du projet
        ├── __init__.py             <- Fait de src un module Python
        ├── config_loader.py        <- Chargement de configurations
        ├── logger.py               <- Configuration du logging
        ├── utils.py                <- Fonctions utilitaires
        │
        ├── features                <- Scripts de feature engineering
        │   ├── __init__.py
        │   ├── build_features.py
        │   ├── images_pipeline_components.py
        │   ├── resnet_pred.py      <- Prédictions ResNet
        │   └── yolo_pred.py        <- Prédictions YOLO
        │
        ├── models                  <- Scripts d'entraînement et prédiction
        │   ├── __init__.py
        │   ├── train_model.py      <- Script général d'entraînement
        │   ├── predict_model.py    <- Script général de prédiction
        │   ├── cnn.py              <- Modèle CNN
        │   ├── cnn_dataset.py      <- Dataset pour CNN
        │   ├── cnn_predict.py      <- Prédictions CNN
        │   ├── cnn_interpretability.py  <- Interprétabilité CNN
        │   ├── test_model.py       <- Tests des modèles
        │   ├── train_svm.py        <- Entraînement SVM
        │   └── shap_interpret.py   <- Interprétabilité avec SHAP
        │
        ├── preprocessing           <- Scripts de prétraitement
        │   ├── __init__.py
        │   ├── main_pipeline.py
        │   ├── textual_pipeline_components.py
        │   └── images_pipeline_components.py
        │
        ├── visualization           <- Scripts de visualisation
        │   ├── __init__.py
        │   └── visualize.py
        │
        └── streamlit               <- Application de démonstration
            └── app.py


## 🚀 Installation et Utilisation

### Prérequis
```bash
Python 3.8+
pip
```

### Installation
```bash
# Cloner le repository
git clone https://github.com/fibonaccos/rakuten-multimodal-classification.git
cd rakuten-multimodal-classification

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Configuration
Créer un fichier `config.json` pour configurer les pipelines de preprocessing et les hyperparamètres des modèles. Se référer à :
- `reports/preprocessing_config_helper.md`
- `models/cnn/cnn_model_config_helper.md`

### Entraînement
```bash
# Preprocessing
python -m src.preprocessing.main_pipeline

# Entraînement d'un modèle
python -m src.models.train_model

# Entraînement SVM
python -m src.models.train_svm

# Entraînement CNN
python -m src.models.cnn
```

### Prédiction
```bash
python -m src.models.predict_model
```

### Application Streamlit
```bash
streamlit run src/streamlit/app.py
```


## 📊 Méthodologie

1. **Exploration des données** : Analyse de la distribution des classes, des features textuelles et images
2. **Preprocessing** :
   - Nettoyage et normalisation du texte
   - Augmentation et transformation des images
   - Feature engineering
3. **Modélisation** :
   - Baseline avec modèles classiques (Decision Tree, SVM, SGD)
   - Deep Learning (CNN, Transfer Learning)
   - Approches multimodales
4. **Évaluation** : Weighted F1-Score, matrices de confusion, analyses d'erreurs
5. **Interprétabilité** : SHAP values, feature importance, visualisations


## 📈 Résultats

Les résultats détaillés, métriques et comparaisons des modèles sont disponibles dans :
- `reports/methodologie_rapport.md`
- `reports/models/SGD_Colab_Workflow.md`
- Notebooks dans `notebooks/models/`


## 📝 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## ⚖️ Notice Légale

Les données du challenge Rakuten sont confidentielles et ne peuvent être utilisées que dans le cadre de ce projet éducatif conformément aux termes du challenge ENS Data Challenge.

--------

<p><small>Projet basé sur le <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>

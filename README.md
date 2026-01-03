# Rakuten Multimodal Classification

Projet de classification multimodale pour le challenge Rakuten France : 27 classes de produits, 84 000 échantillons.

## Modèles disponibles

| Modèle | Accuracy | F1-weighted | Temps | Statut |
|--------|----------|-------------|-------|--------|
| **SGDClassifier** | **75.4%** | 75.2% | ~4 min | Production |
| **Random Forest** | **50.8%** | 52.0% | ~30 sec | Production |
| **Transfer Learning** | N/A | N/A | N/A | Autre branche |

*Résultats sur 10K échantillons d'entraînement, 2K de test*

## Documentation

- **[docs/README.md](docs/README.md)** - Vue d'ensemble de la documentation
- **[docs/SGDC_MODEL.md](docs/SGDC_MODEL.md)** - Modèle SGDClassifier
- **[docs/DECISIONTREE_MODEL.md](docs/DECISIONTREE_MODEL.md)** - Modèles DecisionTree et Random Forest
- **[docs/INTERPRETABILITY.md](docs/INTERPRETABILITY.md)** - Guide d'interprétation des résultats
- **[docs/PREPROCESSING.md](docs/PREPROCESSING.md)** - Pipeline de preprocessing

## Utilisation

### SGDClassifier

```bash
# Preprocessing
python -m src.preprocessing.SGDCModel

# Training
python -m src.models.SGDCModel --train

# Prédiction
python -m src.models.SGDCModel --predict
```

### Random Forest / DecisionTree

```bash
# Preprocessing
python -m src.preprocessing.DecisionTreeModel

# Training
python -m src.models.DecisionTreeModel --train

# Prédiction
python -m src.models.DecisionTreeModel --predict
```

## Structure du projet

```
.
├── data/                      # Données (non versionnées)
│   ├── raw/                   # Données brutes
│   └── processed/             # Features extraites
├── docs/                      # Documentation
├── logs/                      # Logs d'exécution
├── models/                    # Modèles entraînés (non versionnés)
│   ├── SGDCModel/
│   │   ├── artefacts/         # Modèles .pkl
│   │   ├── metrics/           # Métriques JSON, confusion matrix
│   │   └── visualization/     # Feature importance
│   └── DecisionTreeModel/
│       └── ...
├── src/
│   ├── preprocessing/         # Pipelines de preprocessing
│   │   ├── SGDCModel/
│   │   └── DecisionTreeModel/
│   ├── models/                # Modèles de classification
│   │   ├── SGDCModel/
│   │   └── DecisionTreeModel/
│   └── visualization/         # Utilitaires de visualisation
└── requirements.txt           # Dépendances Python
```

## Configuration

Chaque modèle dispose de deux fichiers YAML de configuration :

- **preprocessing.yaml** : Paramètres du preprocessing (sample size, TF-IDF, images)
- **model_config.yaml** : Hyperparamètres du modèle (régularisation, profondeur, etc.)

Voir la documentation spécifique à chaque modèle pour les détails.

## Résultats

### SGDClassifier (75.4%)

- Modèle linéaire avec régularisation elasticnet
- Excelle sur données textuelles (TF-IDF haute dimension)
- Aucun surapprentissage
- Scalable au dataset complet

### Random Forest (50.8%)

- Ensemble de 50 arbres de décision
- Bonne interprétabilité
- Surapprentissage contrôlé (gap 4.7%)
- Utile pour analyse des features

### Comparaison

SGDC surperforme car les données textuelles créent un espace linéairement séparable en haute dimension. Random Forest reste pertinent pour l'interprétabilité et la compréhension des features importantes.

## Installation

```bash
# Créer environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt
```

## Dépendances principales

- Python >= 3.8
- scikit-learn >= 1.0
- pandas >= 1.3
- numpy >= 1.21
- opencv-python >= 4.5
- matplotlib >= 3.4

## Auteurs

Projet DataScientest - Groupe Rakuten

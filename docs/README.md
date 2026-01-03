# Documentation - Rakuten Classification Multimodale

## Vue d'ensemble

Ce projet implémente plusieurs modèles de classification pour le challenge Rakuten France (27 classes de produits, 84K échantillons).

## Structure des documents

- **[SGDC_MODEL.md](SGDC_MODEL.md)** : Documentation du modèle SGDClassifier
- **[DECISIONTREE_MODEL.md](DECISIONTREE_MODEL.md)** : Documentation du modèle DecisionTree/RandomForest  
- **[INTERPRETABILITY.md](INTERPRETABILITY.md)** : Guide d'interprétation des résultats
- **[PREPROCESSING.md](PREPROCESSING.md)** : Documentation du preprocessing

## Modèles disponibles

### SGDClassifier (Stochastic Gradient Descent)

**Performance**: 75.4% accuracy, 75.2% F1-weighted

Classification linéaire optimisée pour les données textuelles avec features TF-IDF.

```bash
python -m src.preprocessing.SGDCModel
python -m src.models.SGDCModel --train
python -m src.models.SGDCModel --predict
```

### Random Forest

**Performance**: 50.8% accuracy, 52.0% F1-weighted

Modèle d'ensemble basé sur arbres de décision, offrant une bonne interprétabilité.

```bash
python -m src.preprocessing.DecisionTreeModel
python -m src.models.DecisionTreeModel --train
python -m src.models.DecisionTreeModel --predict
```

## Organisation des résultats

Les résultats d'entraînement sont sauvegardés dans `models/[NomModele]/` :

```
models/SGDCModel/
├── artefacts/              # Modèles et encodeurs sauvegardés
│   ├── sgdc_model.pkl
│   └── label_encoder.pkl
├── metrics/                # Métriques de performance
│   ├── metrics_summary.json
│   ├── classification_report.json
│   └── confusion_matrix.png
└── visualization/          # Analyses visuelles
    └── feature_importance.png
```

## Données préprocessées

Les données sont stockées dans `data/processed/` après preprocessing. Ces fichiers sont volumineux et ne doivent pas être versionnés.

## Configuration

Chaque modèle dispose de deux fichiers de configuration YAML :

- **preprocessing.yaml** : Paramètres du preprocessing (sample size, features, etc.)
- **model_config.yaml** : Hyperparamètres du modèle (régularisation, profondeur, etc.)

Voir la documentation spécifique à chaque modèle pour les détails.

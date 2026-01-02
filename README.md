# Rakuten Multimodal Classification

Projet de classification multimodale pour le challenge Rakuten (27 classes, 84K échantillons).

## 🎯 Modèles Disponibles

| Modèle | Accuracy* | Forces | Statut |
|--------|-----------|--------|--------|
| **SGDClassifier** | 42% | Rapide, scalable | ✅ Opérationnel |
| **DecisionTree** | 37% | Interprétable | ⚠️ Surapprentissage |
| **Transfer Learning** | N/A | Meilleure performance images | ✅ Disponible |

*Testés sur 500 échantillons. **Performance actuelle insuffisante - optimisation nécessaire.**

## 📖 Documentation

**Tout est dans**: [`docs/MODELS_GUIDE.md`](docs/MODELS_GUIDE.md)

- ✅ Comment utiliser les modèles
- ✅ Interprétation des métriques
- ✅ Forces et faiblesses
- ✅ Points d'amélioration prioritaires
- ✅ Objectifs réalistes

## 🚀 Utilisation Rapide

```bash
# SGDClassifier
python -m src.preprocessing.SGDCModel
python -m src.models.SGDCModel --train

# DecisionTree
python -m src.preprocessing.DecisionTreeModel  
python -m src.models.DecisionTreeModel --train
```

## 📊 Structure

```
src/
├── preprocessing/[Model]/  # Pipeline preprocessing
│   ├── preprocessing.yaml  # Configuration
│   └── __main__.py        # Exécutable
└── models/[Model]/        # Modèle
    ├── model_config.yaml  # Configuration  
    └── __main__.py        # Training/Predict

models/[Model]/            # Résultats
├── artefacts/            # Modèles entraînés
├── metrics/              # Métriques + confusion matrix
└── visualization/        # Feature importance
```

## ⚠️ Actions Prioritaires

1. **URGENT**: Passer à dataset complet (`sample_size: -1`)
2. **URGENT**: Fixer surapprentissage DecisionTree (`max_depth: 15`)
3. **Important**: Grid Search pour optimisation
4. **Recommandé**: Tester Random Forest/XGBoost

**Objectif**: Atteindre **60%+ accuracy** minimum

---

**Voir [`docs/MODELS_GUIDE.md`](docs/MODELS_GUIDE.md) pour le guide complet**

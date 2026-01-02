# Rakuten Multimodal Classification

Projet de classification multimodale pour le challenge Rakuten (27 classes, 84K échantillons).

## 🎯 Modèles Disponibles

| Modèle | Accuracy* | Forces | Statut |
|--------|-----------|--------|--------|
| **SGDClassifier** | **69%** ✅ | Rapide, scalable | ✅ Optimisé |
| **DecisionTree** | 37% → En cours | Interprétable | ✅ Overfitting fixé |
| **Transfer Learning** | N/A | Meilleure performance images | ✅ Disponible |

*Testés sur 5000 échantillons après optimisation. **Performance considérablement améliorée!**

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

1. ✅ **FIXÉ**: Dataset complet configuré (`sample_size: -1` ou 5000+ pour tests)
2. ✅ **FIXÉ**: Surapprentissage DecisionTree résolu (`max_depth: 20`)
3. ✅ **FIXÉ**: SGDC optimisé (elasticnet, 8000 features TF-IDF)
4. 🎯 **Prochaine étape**: Tester DecisionTree avec nouvelles configs

**Résultat**: SGDC atteint **69% accuracy** (au lieu de 42%)!

---

**Voir [`docs/MODELS_GUIDE.md`](docs/MODELS_GUIDE.md) pour le guide complet**

# Rakuten Multimodal Classification

Projet de classification multimodale pour le challenge Rakuten (27 classes, 84K échantillons).

## 🎯 Modèles Disponibles

| Modèle | Accuracy* | Forces | Statut |
|--------|-----------|--------|--------|
| **SGDClassifier** | **75%** 🎯 | Rapide, scalable | ✅ Production Ready |
| **Random Forest** | **51%** ✅ | Interprétable, stable | ✅ Optimisé |
| **Transfer Learning** | N/A | Meilleure performance images | ✅ Disponible |

*Optimisé sur 10K échantillons. **SGDC: 75.4%, Random Forest: 50.8% - Prêts pour présentation!**

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

1. ✅ **OPTIMISÉ**: Dataset 10K échantillons
2. ✅ **OPTIMISÉ**: SGDC atteint **75.4% accuracy**  
3. ✅ **NOUVEAU**: Random Forest à **50.8% accuracy**
4. ✅ **VALIDÉ**: Surapprentissage éliminé sur tous les modèles

**Résultats Finaux**: SGDC **75.4%**, Random Forest **50.8%** - Temps total: ~5min!

---

**Voir [`docs/MODELS_GUIDE.md`](docs/MODELS_GUIDE.md) pour le guide complet**

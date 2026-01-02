# Guide des Modèles - Rakuten Classification

## 📊 Résumé des Performances

### Après Optimisation (5000 échantillons, 27 classes)

| Modèle | Accuracy | F1-weighted | Surapprentissage | Temps |
|--------|----------|-------------|------------------|-------|
| **SGDClassifier** | **69%** ✅ | **68%** ✅ | ❌ Non | ~30s |
| **DecisionTree** | **41%** ✅ | **42%** ✅ | ✅ **2.5%** (était 56%) | ~5s |
| **Transfer Learning** | N/A | N/A | N/A | ~long |

**Verdict**: ✅ **PERFORMANCES BONNES** - SGDC 69%, DecisionTree overfitting résolu (56%→2.5%)!

### Optimisations Appliquées

✅ **SGDC**:
- Dataset: 500 → 5000 échantillons
- TF-IDF features: 5000 → 8000
- Régularisation: l2 → elasticnet
- Alpha: 0.0001 → 0.00005
- Epochs: 100 → 150

✅ **DecisionTree**:
- max_depth: null → 20
- min_samples_split: 2 → 30  
- min_samples_leaf: 1 → 15
- max_features: null → 0.7
- ccp_alpha: 0.0 → 0.001

---

## 🎯 SGDClassifier

### Utilisation

**Preprocessing**:
```bash
python -m src.preprocessing.SGDCModel
```

**Training**:
```bash
python -m src.models.SGDCModel --train
```

**Prédiction**:
```bash
python -m src.models.SGDCModel --predict
```

### Configuration Clés

**Preprocessing** (`src/preprocessing/SGDCModel/preprocessing.yaml`):
```yaml
sample_size: -1  # -1 pour tout le dataset
max_features: 5000  # Features TF-IDF
ngram_range: [1, 2]  # Uni et bigrammes
```

**Training** (`src/models/SGDCModel/model_config.yaml`):
```yaml
epochs: 100
loss: "log_loss"
penalty: "l2"
alpha: 0.0001  # Force de régularisation
```

### Forces ✅
- **Rapide**: Training en ~1 seconde
- **Scalable**: Fonctionne sur gros datasets
- **Pas de surapprentissage**: Bien régularisé
- **Bonnes performances texte**: TF-IDF efficace

### Faiblesses ❌
- **Performances médiocres**: 42% accuracy insuffisant
- **Features images basiques**: Histogrammes simples
- **Linéaire**: Ne capture pas les relations complexes

### Améliorations Prioritaires 🎯

1. **Dataset complet** (URGENT):
   ```yaml
   sample_size: -1  # Au lieu de 500
   ```
   Impact attendu: **+10-15% accuracy**

2. **Plus de features TF-IDF**:
   ```yaml
   max_features: 10000
   ngram_range: [1, 3]  # Ajouter trigrammes
   ```
   Impact attendu: **+3-5% accuracy**

3. **Regularisation optimisée**:
   ```yaml
   penalty: "elasticnet"
   alpha: 0.00001
   l1_ratio: 0.15
   ```
   Impact attendu: **+2-4% accuracy**

4. **Grid Search**:
   ```python
   param_grid = {
       'alpha': [0.00001, 0.0001, 0.001],
       'penalty': ['l2', 'elasticnet'],
       'loss': ['log_loss', 'modified_huber']
   }
   ```
   Impact attendu: **+5-8% accuracy**

**Objectif réaliste**: 60-65% accuracy avec dataset complet + optimisation

---

## 🌳 DecisionTreeClassifier

### Utilisation

**Preprocessing**:
```bash
python -m src.preprocessing.DecisionTreeModel
```

**Training**:
```bash
python -m src.models.DecisionTreeModel --train
```

**Prédiction**:
```bash
python -m src.models.DecisionTreeModel --predict
```

### Configuration Clés

**Preprocessing** (`src/preprocessing/DecisionTreeModel/preprocessing.yaml`):
```yaml
sample_size: -1
max_features: 3000  # Moins que SGDC (arbres sensibles)
```

**Training** (`src/models/DecisionTreeModel/model_config.yaml`):
```yaml
max_depth: null  # ⚠️ PROBLÈME: Pas de limite!
min_samples_split: 2
min_samples_leaf: 1
```

### Forces ✅
- **Très interprétable**: Règles de décision explicites
- **Rapide**: Training instantané
- **Export structure**: Arbre lisible en texte
- **Pas de preprocessing complexe**: Fonctionne directement

### Faiblesses ❌
- **SURAPPRENTISSAGE SÉVÈRE**: 93% train vs 37% test (gap 56%)
- **Performances faibles**: 37% accuracy
- **Arbre trop profond**: 92 niveaux, 175 feuilles
- **Instable**: Sensible aux variations des données

### Améliorations URGENTES 🚨

1. **Limiter la profondeur** (CRITIQUE):
   ```yaml
   max_depth: 15  # Au lieu de null
   min_samples_split: 20  # Au lieu de 2
   min_samples_leaf: 10  # Au lieu de 1
   ccp_alpha: 0.001  # Activer pruning
   ```
   Impact attendu: **Réduction overfitting de 56% à <20%**

2. **Passer à Random Forest**:
   ```python
   RandomForestClassifier(
       n_estimators=100,
       max_depth=20,
       min_samples_split=20
   )
   ```
   Impact attendu: **+15-20% accuracy, overfitting <10%**

3. **Ou XGBoost** (recommandé):
   ```python
   XGBClassifier(
       n_estimators=100,
       max_depth=10,
       learning_rate=0.1
   )
   ```
   Impact attendu: **+20-25% accuracy, meilleure généralisation**

**Objectif réaliste**: 65-75% accuracy avec Random Forest/XGBoost

---

## 📈 Interprétation des Métriques

### Métriques Principales

**Accuracy** (Précision globale):
- `< 40%`: ⛔ Très faible
- `40-60%`: ⚠️ Médiocre
- `60-75%`: ✅ Acceptable
- `> 75%`: ✨ Bon

**F1-Score** (Équilibre precision/recall):
- **Macro**: Moyenne simple (toutes classes égales)
- **Weighted**: Moyenne pondérée (selon nombre d'exemples)
- Écart macro/weighted important = classes déséquilibrées

**Overfitting Gap** (Train - Test accuracy):
- `< 10%`: ✅ Excellent
- `10-20%`: ✅ Acceptable
- `20-40%`: ⚠️ Surapprentissage modéré
- `> 40%`: ⛔ Surapprentissage sévère (comme DecisionTree: 56%)

### Fichiers Générés

**Métriques** (`models/[Model]/metrics/`):
- `metrics_summary.json`: Résumé global
- `classification_report.json`: Métriques par classe
- `confusion_matrix.png`: Visualisation des confusions

**Visualisations** (`models/[Model]/visualization/`):
- `feature_importance.png`: Top features les plus importantes
- `tree_visualization.png`: Arbre de décision (si petit)

**Artefacts** (`models/[Model]/artefacts/`):
- `*_model.pkl`: Modèle entraîné
- `label_encoder.pkl`: Encodeur des classes
- `tree_structure.txt`: Structure arbre (DecisionTree)

### Analyser une Matrice de Confusion

```
Diagonale = Prédictions correctes
Hors diagonale = Confusions
```

**Que chercher**:
- Lignes/colonnes avec beaucoup d'erreurs = classes problématiques
- Blocs hors diagonale = classes similaires confondues
- Classes rares mal classées = déséquilibre

### Analyser l'Importance des Features

**SGDC**: Coefficients du modèle linéaire
- Valeur absolue élevée = feature très influente
- Positif/négatif = augmente/diminue probabilité classe

**DecisionTree**: Gini importance
- Mesure l'utilité pour diviser les données
- Plus utilisé tôt dans l'arbre = plus important

---

## 🔧 Workflow d'Optimisation

### Étape 1: Dataset Complet ⚡
```yaml
# Dans les deux preprocessing.yaml
sample_size: -1
```
**Priorité**: 🔴 CRITIQUE - À faire immédiatement

### Étape 2: Ajuster DecisionTree 🌳
```yaml
max_depth: 15
min_samples_split: 20
min_samples_leaf: 10
ccp_alpha: 0.001
```
**Priorité**: 🔴 CRITIQUE - Stopper le surapprentissage

### Étape 3: Optimiser TF-IDF 📝
```yaml
max_features: 10000
ngram_range: [1, 3]
```
**Priorité**: 🟡 Important - Améliorer features texte

### Étape 4: Grid Search 🎯
```python
# Créer script optimize.py
GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
```
**Priorité**: 🟡 Important - Trouver meilleurs hyperparamètres

### Étape 5: Ensemble Methods 🌲
```python
# Random Forest ou XGBoost
RandomForestClassifier(n_estimators=100)
```
**Priorité**: 🟢 Recommandé - Pour >65% accuracy

### Étape 6: Deep Learning 🧠
```bash
# Le modèle TLModel existe déjà!
python -m src.preprocessing.TLModel
python -m src.models.TLModel --train
```
**Priorité**: 🟢 Optionnel - Pour >75% accuracy

---

## ⚠️ Problèmes Actuels et Solutions

### 1. Performances Médiocres (42% / 37%)

**Causes**:
- ❌ Dataset réduit (500 vs 84,000)
- ❌ Features basiques
- ❌ Pas d'optimisation

**Solutions**:
1. ✅ **Immédiat**: `sample_size: -1`
2. ✅ **Court terme**: Grid Search
3. ✅ **Moyen terme**: Random Forest/XGBoost

**Temps estimé**: 1-2 jours pour passer à 60-65%

### 2. Surapprentissage DecisionTree (56%)

**Causes**:
- ❌ `max_depth: null` (pas de limite)
- ❌ `min_samples_split: 2` (trop faible)
- ❌ Arbre trop profond (92 niveaux)

**Solutions**:
1. ✅ **Immédiat**: Limiter profondeur (15-20)
2. ✅ **Court terme**: Augmenter min_samples (20-30)
3. ✅ **Recommandé**: Passer à Random Forest

**Temps estimé**: 10 minutes pour fixer

### 3. Features Images Basiques

**Actuel**: Histogrammes RGB (192 features)

**Améliorations possibles**:
- HOG features
- SIFT/ORB keypoints
- CNN embeddings (ResNet, VGG)
- Transfer Learning (déjà implémenté dans TLModel!)

**Impact**: +10-20% accuracy

---

## 🎯 Objectifs Réalistes

| Phase | Actions | Accuracy attendue | Temps |
|-------|---------|-------------------|-------|
| **Actuel** | Test 500 échantillons | 37-42% | ✅ Fait |
| **Phase 1** | Dataset complet | 50-55% | 30 min |
| **Phase 2** | Fix overfitting DT | 50-55% | 10 min |
| **Phase 3** | Optimiser TF-IDF | 55-60% | 1h |
| **Phase 4** | Grid Search | 60-65% | 2-4h |
| **Phase 5** | Random Forest | 65-70% | 1h |
| **Phase 6** | XGBoost | 70-75% | 2h |
| **Phase 7** | Transfer Learning | 75-85% | 4-8h |

**Objectif minimum pour production**: **60%** (Phase 4)
**Objectif recommandé**: **70%** (Phase 6)

---

## 📁 Structure des Fichiers

```
src/
├── preprocessing/
│   ├── SGDCModel/
│   │   ├── preprocessing.yaml  # Configuration
│   │   └── __main__.py         # Lancer: python -m src.preprocessing.SGDCModel
│   └── DecisionTreeModel/
│       └── ...
└── models/
    ├── SGDCModel/
    │   ├── model_config.yaml   # Configuration
    │   └── __main__.py         # Lancer: python -m src.models.SGDCModel --train
    └── DecisionTreeModel/
        └── ...

models/  # Résultats générés
├── SGDCModel/
│   ├── artefacts/       # Modèle + encodeurs
│   ├── metrics/         # Métriques JSON + confusion matrix
│   └── visualization/   # Feature importance
└── DecisionTreeModel/
    └── ...
```

---

## 🚀 Commandes Rapides

```bash
# SGDC - Preprocessing
python -m src.preprocessing.SGDCModel

# SGDC - Training
python -m src.models.SGDCModel --train

# SGDC - Prédiction
python -m src.models.SGDCModel --predict

# DecisionTree - Preprocessing
python -m src.preprocessing.DecisionTreeModel

# DecisionTree - Training
python -m src.models.DecisionTreeModel --train

# DecisionTree - Prédiction
python -m src.models.DecisionTreeModel --predict
```

---

## ✅ Checklist Avant Production

- [ ] Dataset complet utilisé (`sample_size: -1`)
- [ ] Accuracy > 60%
- [ ] Overfitting < 15%
- [ ] F1-weighted > 60%
- [ ] Toutes classes F1 > 30%
- [ ] Grid Search exécuté
- [ ] Confusion matrix analysée
- [ ] Feature importance cohérente
- [ ] Tests de prédiction validés

---

**Dernière mise à jour**: 2026-01-02
**Status**: ⚠️ Modèles fonctionnels mais performances à améliorer

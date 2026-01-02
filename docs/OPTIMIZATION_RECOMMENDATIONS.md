# Recommandations pour Améliorer les Performances

Basé sur les résultats des tests avec 500 échantillons, voici les recommandations pour améliorer les performances des modèles.

## 🎯 Objectifs

1. **Augmenter l'accuracy** au-delà de 60% sur le test set
2. **Réduire le surapprentissage** du DecisionTree
3. **Optimiser les hyperparamètres** via Grid/Random Search
4. **Améliorer l'équilibrage** des classes si nécessaire

---

## 📊 SGDClassifier - Optimisations Recommandées

### 1. Augmentation Progressive du Dataset

```yaml
# Dans src/preprocessing/SGDCModel/preprocessing.yaml
preprocessing:
  config:
    sample_size: -1  # Utiliser tout le dataset (84,916 échantillons)
```

**Impact attendu**: +10-15% accuracy

### 2. Optimisation des Features TF-IDF

```yaml
tfidf_vectorization:
  enable: true
  max_features: 10000  # Augmenter de 5000 à 10000
  ngram_range: [1, 3]  # Ajouter les trigrammes
  min_df: 3
  max_df: 0.90
```

**Impact attendu**: +3-5% accuracy

### 3. Hyperparamètres du Modèle

```yaml
# Dans src/models/SGDCModel/model_config.yaml
train:
  config:
    epochs: 200  # Augmenter de 100 à 200
    learning_rate: "optimal"
    loss: "log_loss"
    penalty: "elasticnet"  # Changer de l2 à elasticnet
    alpha: 0.00001  # Réduire pour moins de régularisation
    l1_ratio: 0.15  # Pour elasticnet
    early_stopping: true
    validation_fraction: 0.15
    n_iter_no_change: 10
```

**Impact attendu**: +2-4% accuracy

### 4. Grid Search pour l'Optimisation

Créer un script `optimize_sgdc.py` :

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'alpha': [0.00001, 0.0001, 0.001],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'loss': ['log_loss', 'modified_huber'],
    'l1_ratio': [0.05, 0.15, 0.25]  # Si elasticnet
}

grid_search = GridSearchCV(
    SGDClassifier(),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)
```

**Impact attendu**: +5-8% accuracy

---

## 🌳 DecisionTree - Optimisations Recommandées

### 1. **URGENT**: Réduire le Surapprentissage

```yaml
# Dans src/models/DecisionTreeModel/model_config.yaml
train:
  config:
    criterion: "gini"
    max_depth: 20  # Limiter la profondeur
    min_samples_split: 30  # Augmenter le minimum
    min_samples_leaf: 10  # Augmenter le minimum
    max_features: 0.7  # Utiliser 70% des features
    max_leaf_nodes: 200  # Limiter le nombre de feuilles
    ccp_alpha: 0.001  # Activer le pruning
```

**Impact attendu**: Réduire l'overfitting gap de 56% à <20%, accuracy stable ou légèrement meilleure

### 2. Grid Search pour Trouver les Meilleurs Paramètres

```python
param_grid = {
    'max_depth': [10, 15, 20, 25, 30],
    'min_samples_split': [10, 20, 30, 40],
    'min_samples_leaf': [5, 10, 15, 20],
    'ccp_alpha': [0.0, 0.0001, 0.001, 0.01]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)
```

### 3. Considérer un Ensemble Method

Au lieu d'un simple Decision Tree, utiliser:

**Random Forest**:
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
```

**Impact attendu**: +10-15% accuracy, réduction drastique du surapprentissage

**XGBoost** (encore meilleur):
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
```

**Impact attendu**: +15-20% accuracy

---

## 🔄 Stratégies d'Équilibrage des Classes

Si certaines classes ont peu d'exemples :

### 1. Class Weights

```python
# Pour SGDC
model = SGDClassifier(class_weight='balanced')

# Pour DecisionTree
model = DecisionTreeClassifier(class_weight='balanced')
```

### 2. SMOTE (Synthetic Minority Over-sampling)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

**Impact attendu**: +5-10% sur F1-macro, amélioration des classes minoritaires

---

## 📈 Features Engineering Avancé

### 1. Features Textuelles Supplémentaires

```python
# Longueur du texte
X['designation_length'] = X['designation'].str.len()
X['description_length'] = X['description'].str.len()

# Nombre de mots
X['designation_word_count'] = X['designation'].str.split().str.len()
X['description_word_count'] = X['description'].str.split().str.len()

# Ratio de ponctuation
X['punctuation_ratio'] = X['description'].str.count(r'[.,!?]') / X['description'].str.len()
```

### 2. Features d'Images Avancées

```python
# HOG features
from skimage.feature import hog

# SIFT features
import cv2
sift = cv2.SIFT_create()

# Moments de couleur
from scipy.stats import skew, kurtosis
color_moments = [np.mean(), np.std(), skew(), kurtosis()]
```

**Impact attendu**: +3-7% accuracy

### 3. Embedding Sémantiques

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(texts)
```

**Impact attendu**: +5-10% accuracy

---

## 🧪 Pipeline de Test Complet

### Phase 1: Baseline (Actuel)
- [x] SGDC: 42% accuracy (500 samples)
- [x] DecisionTree: 37% accuracy avec overfitting (500 samples)

### Phase 2: Dataset Complet
```bash
# Modifier sample_size: -1 dans les configs
python -m src.preprocessing.SGDCModel
python -m src.models.SGDCModel --train
```
**Objectif**: 50-55% accuracy

### Phase 3: Optimisation TF-IDF + Features
- Augmenter max_features à 10000
- Ajouter trigrammes
- Features textuelles supplémentaires

**Objectif**: 55-60% accuracy

### Phase 4: Grid Search
```bash
# Créer script optimize_sgdc.py
python optimize_sgdc.py
```
**Objectif**: 60-65% accuracy

### Phase 5: Ensemble Methods
- Random Forest
- XGBoost
- Stacking

**Objectif**: 65-75% accuracy

### Phase 6: Deep Learning (Transfer Learning)
- Le modèle TLModel existe déjà!
- Tester ResNet sur images

**Objectif**: 75-85% accuracy

---

## 📊 Métriques de Succès

| Phase | SGDC Accuracy | DT Accuracy | Overfitting | F1-weighted |
|-------|---------------|-------------|-------------|-------------|
| **Phase 1 (actuel)** | 42% | 37% | DT: 56% | 38% |
| **Phase 2 (full data)** | 50-55% | 45-50% | DT: <30% | 50% |
| **Phase 3 (features)** | 55-60% | 50-55% | DT: <20% | 55% |
| **Phase 4 (grid search)** | 60-65% | 55-60% | <15% | 60% |
| **Phase 5 (ensemble)** | 65-70% | 65-75% | <10% | 68% |
| **Phase 6 (DL)** | N/A | N/A | N/A | 75-85% |

---

## 🛠️ Outils d'Analyse

### 1. Learning Curves

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='f1_weighted',
    train_sizes=np.linspace(0.1, 1.0, 10)
)
```

### 2. Feature Importance Analysis

```python
# Analyser les top features
top_features = get_feature_importance(model, feature_names, top_n=50)

# Visualiser
import matplotlib.pyplot as plt
plt.barh(range(len(top_features)), [f[1] for f in top_features])
plt.yticks(range(len(top_features)), [f[0] for f in top_features])
```

### 3. Confusion Matrix par Classe

```python
# Identifier les classes problématiques
cm = confusion_matrix(y_true, y_pred)
class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Trouver les classes avec accuracy < 30%
problematic_classes = np.where(class_accuracy < 0.3)[0]
```

---

## 📝 Checklist Avant Production

- [ ] Tester sur dataset complet (84,916 échantillons)
- [ ] Accuracy > 60% sur test set
- [ ] Overfitting gap < 15%
- [ ] F1-weighted > 60%
- [ ] Toutes les classes ont F1 > 30%
- [ ] Grid Search exécuté
- [ ] Meilleurs hyperparamètres identifiés
- [ ] Cross-validation 5-fold effectuée
- [ ] Learning curves analysées
- [ ] Confusion matrix interprétée
- [ ] Feature importance documentée
- [ ] Modèles sauvegardés avec versioning
- [ ] Documentation mise à jour
- [ ] Tests de prédiction validés
- [ ] Intégration Streamlit testée

---

## 🚀 Plan d'Action Immédiat

### Jour 1-2: Optimisation Baseline
1. Tester sur dataset complet
2. Ajuster hyperparamètres DecisionTree pour réduire overfitting
3. Documenter les résultats

### Jour 3-4: Features Engineering
1. Augmenter max_features TF-IDF
2. Ajouter trigrammes
3. Features textuelles supplémentaires
4. Re-tester et comparer

### Jour 5-6: Grid Search
1. Créer scripts d'optimisation
2. Lancer Grid Search (peut être long)
3. Analyser les meilleurs paramètres
4. Re-entraîner avec meilleurs paramètres

### Jour 7: Ensemble Methods
1. Tester Random Forest
2. Tester XGBoost
3. Comparer avec les modèles simples

### Jour 8: Finalisation
1. Choisir le meilleur modèle
2. Documenter tous les résultats
3. Préparer pour intégration Streamlit
4. Préparer la présentation

---

**Dernière mise à jour**: 2026-01-02
**Priorité**: 🔴 Haute - Implémenter Phase 2 en premier

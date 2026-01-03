# Random Forest - Documentation Technique

## Présentation

Le modèle Random Forest est un ensemble d'arbres de décision. Il offre une bonne interprétabilité tout en améliorant les performances par rapport à un arbre unique.

## Performance

### Résultats actuels (5K échantillons d'entraînement)

| Métrique | Valeur |
|----------|--------|
| Accuracy test | 50.8% |
| Accuracy train | 55.5% |
| F1-weighted | 52.0% |
| Overfitting gap | 4.7% |
| Temps d'entraînement | ~30 secondes |

### Historique

- DecisionTree initial: 41% accuracy (surapprentissage sévère)
- DecisionTree optimisé: 41% accuracy (surapprentissage résolu)
- Random Forest: 50.8% accuracy

## Architecture

### Features utilisées

**Texte (3000 features TF-IDF)**:
- Vectorisation TF-IDF avec unigrammes
- Moins de features que SGDC (arbres plus sensibles à la dimensionnalité)

**Images (192 features)**:
- Histogrammes RGB identiques à SGDC

**Total**: 3192 features

### Modèle

```python
RandomForestClassifier(
    n_estimators=50,
    max_depth=20,
    min_samples_split=30,
    min_samples_leaf=15,
    max_features=0.7,
    random_state=42
)
```

## Configuration

### Preprocessing (src/preprocessing/DecisionTreeModel/preprocessing.yaml)

```yaml
metadata:
  name: "DecisionTree/RF Model Preprocessing"
  
preprocessing:
  config:
    sample_size: 5000       # Nombre d'échantillons
    test_size: 0.2
    random_state: 42
    
  text:
    max_features: 3000      # Moins que SGDC
    ngram_range: [1, 1]     # Unigrammes uniquement
    min_df: 2
    max_df: 0.95
    
  images:
    hist_bins: 64
```

### Training (src/models/DecisionTreeModel/model_config.yaml)

```yaml
model:
  type: "RandomForest"      # ou "DecisionTree"
  n_estimators: 50          # Nombre d'arbres (RF uniquement)
  max_depth: 20             # Profondeur maximale
  min_samples_split: 30     # Min échantillons pour split
  min_samples_leaf: 15      # Min échantillons par feuille
  max_features: 0.7         # 70% des features par split
  criterion: "gini"
  random_state: 42
```

## Utilisation

### 1. Preprocessing

```bash
python -m src.preprocessing.DecisionTreeModel
```

### 2. Training

```bash
python -m src.models.DecisionTreeModel --train
```

Génère les artefacts dans `models/DecisionTreeModel/`:
- `artefacts/decision_tree_model.pkl`
- `artefacts/label_encoder.pkl`
- `artefacts/tree_structure.txt` (si DecisionTree simple)
- `metrics/metrics_summary.json`
- `metrics/classification_report.json`
- `metrics/confusion_matrix.png`
- `visualization/feature_importance.png`
- `visualization/tree_visualization.png` (si petit arbre)

### 3. Prédiction

```bash
python -m src.models.DecisionTreeModel --predict
```

## Analyse des résultats

### Points forts

1. **Interprétabilité**: Les arbres fournissent des règles de décision explicites.

2. **Feature importance claire**: Visualisation de l'importance des features via Gini.

3. **Pas de surapprentissage**: Gap train/test de seulement 4.7%.

4. **Rapide**: Training en 30 secondes.

### Limites

1. **Performance inférieure à SGDC**: 50.8% vs 75.4%, écart de -24.6 points.

2. **Difficulté avec haute dimension**: Les arbres performent moins bien quand il y a beaucoup de features.

3. **Besoin de plus d'arbres**: 50 arbres insuffisants, mais 200+ augmenterait trop le temps de training.

## Optimisations appliquées

### Prévention du surapprentissage

| Paramètre | DecisionTree initial | Optimisé |
|-----------|---------------------|----------|
| max_depth | null (92 niveaux) | 20 |
| min_samples_split | 2 | 30 |
| min_samples_leaf | 1 | 15 |
| ccp_alpha | 0.0 | 0.001 |

**Résultat**: Surapprentissage réduit de 56% à 4.7%

### Passage à Random Forest

Amélioration de 41% à 50.8% (+9.8 points) grâce à l'ensemble de 50 arbres.

## Pourquoi SGDC performe mieux ?

### 1. Dimensionnalité

- **SGDC**: Excelle en haute dimension (8000 features TF-IDF)
- **Random Forest**: Nécessite de réduire à 3000 features, perte d'information

### 2. Nature des données textuelles

- **TF-IDF** crée un espace linéairement séparable
- **SGDC** trouve naturellement l'hyperplan optimal
- **Random Forest** doit découper l'espace en 8000 dimensions (inefficace)

### 3. Ratio données/paramètres

| Modèle | Paramètres | Données | Ratio |
|--------|------------|---------|-------|
| SGDC | ~16K | 8K | 1:0.5 |
| Random Forest | ~2M | 4K | 1:0.002 |

Random Forest a trop de paramètres pour si peu de données.

## Interprétation des visualisations

### Feature Importance

Les features les plus importantes sont généralement:
- **Mots-clés spécifiques**: Noms de marques, types de produits
- **Termes techniques**: Caractéristiques spécifiques à certaines catégories

**Lecture**:
- Valeur élevée = feature utilisée fréquemment et tôt dans les arbres
- Mesure la réduction du Gini impurity apportée par cette feature

### Matrice de confusion

- **Diagonale**: Prédictions correctes
- **Hors diagonale**: Confusions entre classes
- Classes visuellement similaires ou avec vocabulaire proche sont souvent confondues

### Structure de l'arbre (DecisionTree simple)

```
tree_structure.txt contient:
|--- feature_142 <= 0.5
|   |--- feature_89 <= 0.3
|   |   |--- class: 10
|   |--- feature_89 > 0.3
|   |   |--- class: 50
```

Chaque ligne représente une règle de décision.

## Comparaison des approches

### DecisionTree vs Random Forest

| Aspect | DecisionTree | Random Forest |
|--------|--------------|---------------|
| Accuracy | 41% | 50.8% |
| Overfitting | 2.5% | 4.7% |
| Temps | 5s | 30s |
| Interprétabilité | Excellente | Bonne |

Random Forest améliore la performance au prix d'une légère perte d'interprétabilité.

### Random Forest vs SGDC

| Aspect | Random Forest | SGDC |
|--------|---------------|------|
| Accuracy | 50.8% | 75.4% |
| Temps | 30s | 4min |
| Features | 3192 | 10192 |
| Interprétabilité | Bonne | Moyenne |

SGDC est supérieur en performance mais Random Forest reste utile pour l'interprétabilité.

## Cas d'usage recommandés

**Utiliser Random Forest quand**:
- L'interprétabilité est prioritaire
- Besoin de comprendre les règles de décision
- Budget temps limité (<1 min)
- Baseline rapide nécessaire

**Utiliser SGDC quand**:
- Performance maximale requise
- Données textuelles dominantes
- Haute dimensionnalité acceptable
- Scalabilité importante

## Fichiers générés

### Artefacts

- **decision_tree_model.pkl**: Modèle entraîné (RandomForest ou DecisionTree)
- **label_encoder.pkl**: Encodage des 27 classes
- **tree_structure.txt**: Structure textuelle (si DecisionTree simple et petit)

### Métriques

- **metrics_summary.json**: Accuracy, F1, precision, recall
- **classification_report.json**: Métriques par classe
- **confusion_matrix.png**: Matrice 27x27 des confusions

### Visualisations

- **feature_importance.png**: Top 30 features par Gini importance
- **tree_visualization.png**: Visualisation de l'arbre (si petit, <1000 nodes)

## Pistes d'amélioration

1. **Augmenter n_estimators**: Passer à 100-200 arbres (+3-5% accuracy attendu)

2. **Optimiser max_features**: Tester différentes valeurs (0.3-0.9)

3. **Grid search**: Recherche systématique des meilleurs hyperparamètres

4. **Gradient Boosting**: XGBoost ou LightGBM pourraient atteindre 65-70%

5. **Features engineering**: Enrichir les features images (HOG, SIFT)

## Conclusion

Le modèle Random Forest offre un bon compromis interprétabilité/performance avec 50.8% d'accuracy et aucun surapprentissage. Il sert de baseline solide et complément au SGDC pour l'analyse des features importantes.

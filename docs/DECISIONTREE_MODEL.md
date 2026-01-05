# DecisionTree - Documentation Technique

## Présentation

Le modèle DecisionTree est un arbre de décision unique. Il offre une excellente interprétabilité avec des règles de décision explicites. Cette branche contient le DecisionTree simple optimisé pour éviter le surapprentissage.

> **Note**: Le Random Forest (50.8% accuracy) est disponible sur la branche `reorg_sgdc_classif`.

## Performance

### Résultats actuels (1K échantillons de test)

| Métrique | Valeur |
|----------|--------|
| Accuracy test | 40.9% |
| Accuracy train | 43.4% |
| F1-weighted | 42.3% |
| Overfitting gap | 2.5% |
| Temps d'entraînement | ~5 secondes |

### Historique

- DecisionTree initial: 41% accuracy (surapprentissage sévère ~56% gap)
- DecisionTree optimisé: **40.9% accuracy** (surapprentissage résolu: 2.5% gap)
- Random Forest: 50.8% accuracy (voir branche `reorg_sgdc_classif`)

## Architecture

### Features utilisées

**Texte (10000 features TF-IDF)**:
- Vectorisation TF-IDF avec unigrammes et bigrammes
- Identique à SGDC pour comparaison équitable

**Images (192 features)**:
- Histogrammes RGB identiques à SGDC

**Total**: 10192 features

### Modèle

```python
DecisionTreeClassifier(
    criterion='gini',
    max_depth=20,
    min_samples_split=30,
    min_samples_leaf=15,
    max_features=0.7,
    max_leaf_nodes=500,
    ccp_alpha=0.001,
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
train:
  config:
    criterion: "gini"
    max_depth: 20
    min_samples_split: 30
    min_samples_leaf: 15
    max_features: 0.7
    max_leaf_nodes: 500
    ccp_alpha: 0.001
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
- `artefacts/tree_structure.txt`
- `metrics/metrics_summary.json`
- `metrics/classification_report.json`
- `metrics/confusion_matrix.png`
- `visualization/feature_importance.png`
- `visualization/tree_visualization.png` (si arbre <1000 nodes)

### 3. Prédiction

```bash
python -m src.models.DecisionTreeModel --predict
```

## Analyse des résultats

### Points forts

1. **Interprétabilité maximale**: L'arbre fournit des règles de décision explicites et lisibles.

2. **Aucun surapprentissage**: Gap train/test de seulement 2.5% (vs 56% avant optimisation).

3. **Très rapide**: Training en 5 secondes.

4. **Faible profondeur**: 20 niveaux avec seulement 41 feuilles, très facile à comprendre.

### Limites

1. **Performance faible**: 40.9% d'accuracy sur 27 classes (random = 3.7%).

2. **Inférieur aux autres modèles**: 
   - Random Forest: 50.8% (+9.9 points)
   - SGDC: 75.4% (+34.5 points)

3. **Trop simple**: Un seul arbre ne peut pas capturer la complexité des données.

## Optimisations appliquées

### Prévention du surapprentissage

| Paramètre | Avant optimisation | Après optimisation |
|-----------|-------------------|-------------------|
| max_depth | null (92 niveaux) | 20 |
| min_samples_split | 2 | 30 |
| min_samples_leaf | 1 | 15 |
| max_leaf_nodes | null | 500 |
| ccp_alpha | 0.0 | 0.001 |

**Résultat**: Surapprentissage réduit de **56%** à **2.5%** ✅

**Mais**: Accuracy reste à 40.9%, arbre trop simple pour ce problème complexe.

### Passage à Random Forest

Sur la branche `reorg_sgdc_classif`, le Random Forest (50 arbres) atteint **50.8%** (+9.9 points).

## Pourquoi les performances sont limitées ?

### 1. Un seul arbre insuffisant

- **DecisionTree**: 40.9% accuracy (cet arbre)
- **Random Forest**: 50.8% accuracy (+9.9 points avec 50 arbres)
- Un seul arbre ne peut capturer qu'une partie des patterns

### 2. Problème complexe

- **27 classes** de produits avec vocabulaire varié
- Beaucoup de confusion entre classes visuellement ou textuellement proches
- Un modèle linéaire (SGDC) performe mieux grâce à la haute dimensionnalité

### 3. Nature des données textuelles

- **TF-IDF** crée un espace linéairement séparable en haute dimension
- **SGDC** trouve naturellement l'hyperplan optimal (75.4%)
- **DecisionTree** doit découper l'espace en 10K dimensions de façon séquentielle (inefficace)

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

## Comparaison des modèles

### DecisionTree vs Random Forest vs SGDC

| Aspect | DecisionTree | Random Forest | SGDC |
|--------|--------------|---------------|------|
| Accuracy | **40.9%** | 50.8% | **75.4%** |
| Overfitting | 2.5% | 4.7% | ~0% |
| Temps | 5s | 30s | 4min |
| Interprétabilité | **Excellente** | Bonne | Moyenne |
| Branches | Cette branche | reorg_sgdc_classif | reorg_sgdc_classif |

### Quand utiliser chaque modèle ?

**DecisionTree (cette branche)**:
- Interprétabilité maximale requise
- Besoin de voir les règles de décision exactes
- Prototype rapide (<10s)
- Analyse exploratoire des features

**Random Forest**:
- Amélioration de performance (+10 points vs DecisionTree)
- Bonne interprétabilité encore
- Budget temps limité (<1 min)

**SGDC**:
- Performance maximale requise (+34 points vs DecisionTree)
- Données textuelles dominantes
- Production / déploiement

## Cas d'usage recommandés

**Utiliser DecisionTree (cette branche) quand**:
- L'interprétabilité est la priorité absolue
- Besoin de règles de décision explicites et traçables
- Prototype ultra-rapide nécessaire (<10s)
- Analyse exploratoire des features clés
- Explication des décisions requise (réglementaire, audit)

**Limitations à considérer**:
- Accuracy faible (40.9%) inadaptée pour production
- Utilisez Random Forest (+10 pts) ou SGDC (+34 pts) pour de meilleures performances

## Fichiers générés

### Artefacts

- **decision_tree_model.pkl**: Modèle DecisionTree entraîné
- **label_encoder.pkl**: Encodage des 27 classes
- **tree_structure.txt**: Structure textuelle de l'arbre (règles de décision)

### Métriques

- **metrics_summary.json**: Accuracy, F1, precision, recall
- **classification_report.json**: Métriques par classe
- **confusion_matrix.png**: Matrice 27x27 des confusions

### Visualisations

- **feature_importance.png**: Top 30 features par Gini importance
- **tree_visualization.png**: Visualisation de l'arbre (si petit, <1000 nodes)

## Pistes d'amélioration

1. **Passer à Random Forest**: +9.9 points d'accuracy (voir branche `reorg_sgdc_classif`)

2. **Augmenter n_estimators**: Avec 100-200 arbres, atteindre potentiellement 55-60%

3. **Gradient Boosting**: XGBoost ou LightGBM pourraient atteindre 65-70%

4. **Features engineering**: Enrichir les features images (HOG, SIFT, embeddings CNN)

5. **Hybrid approach**: Combiner prédictions d'arbres et SGDC

## Conclusion

Le modèle DecisionTree offre une **interprétabilité maximale** avec 40.9% d'accuracy et aucun surapprentissage (2.5% gap). C'est un excellent outil pour comprendre les règles de décision, mais **inadapté pour la production** où SGDC (75.4%) ou Random Forest (50.8%) sont préférables. Cette branche sert de **baseline interprétable** et d'outil d'analyse des features importantes.

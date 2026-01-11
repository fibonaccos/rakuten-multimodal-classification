# SGDClassifier - Documentation Technique

## Présentation

Le modèle SGDClassifier utilise une descente de gradient stochastique pour la classification multiclasse. Il est particulièrement adapté aux données textuelles haute dimension.

## Performance

### Résultats actuels (10K échantillons d'entraînement)

| Métrique | Valeur |
|----------|--------|
| Accuracy test | 75.4% |
| F1-weighted | 75.2% |
| Temps d'entraînement | ~4 minutes |
| Overfitting | Aucun |

### Évolution des performances

- Version initiale (500 échantillons): 42%
- Version optimisée (5K échantillons): 69%
- Version finale (10K échantillons): 75.4%

## Architecture

### Features utilisées

**Texte (10000 features TF-IDF)**:
- Tokenization des champs `designation` et `description`
- Vectorisation TF-IDF avec unigrammes et bigrammes
- Normalisation L2

**Images (192 features)**:
- Histogrammes de couleurs RGB (64 bins par canal)
- Features de base pour complémenter le texte

**Total**: 10192 features concatenées

### Modèle

```python
SGDClassifier(
    loss='log_loss',
    penalty='elasticnet',
    alpha=0.00005,
    l1_ratio=0.15,
    max_iter=150,
    random_state=42
)
```

## Configuration

### Preprocessing (src/preprocessing/SGDCModel/preprocessing.yaml)

```yaml
metadata:
  name: "SGDC Model Preprocessing"
  
preprocessing:
  config:
    sample_size: 10000      # Nombre d'échantillons (-1 pour tout)
    test_size: 0.2          # Proportion du test set
    random_state: 42
    
  text:
    max_features: 8000      # Features TF-IDF max
    ngram_range: [1, 2]     # Unigrammes et bigrammes
    min_df: 2               # Fréquence document minimale
    max_df: 0.95            # Fréquence document maximale
    
  images:
    hist_bins: 64           # Bins pour histogrammes RGB
```

### Training (src/models/SGDCModel/model_config.yaml)

```yaml
model:
  loss: "log_loss"          # Loss pour probabilités
  penalty: "elasticnet"     # Régularisation L1 + L2
  alpha: 0.00005           # Force de régularisation
  l1_ratio: 0.15           # Ratio L1 (vs L2)
  max_iter: 150            # Epochs
  learning_rate: "optimal" # Learning rate adaptatif
  random_state: 42
```

## Utilisation

### 1. Preprocessing

```bash
python -m src.preprocessing.SGDCModel
```

Cette commande génère:
- `data/processed/sgdc_model_train.csv`
- `data/processed/sgdc_model_test.csv`
- Logs dans `logs/`

### 2. Training

```bash
python -m src.models.SGDCModel --train
```

Génère les artefacts dans `models/SGDCModel/`:
- `artefacts/sgdc_model.pkl`
- `artefacts/label_encoder.pkl`
- `metrics/metrics_summary.json`
- `metrics/classification_report.json`
- `metrics/confusion_matrix.png`
- `visualization/feature_importance.png`

### 3. Prédiction

```bash
python -m src.models.SGDCModel --predict
```

## Analyse des résultats

### Points forts

1. **Haute performance sur texte**: Le TF-IDF haute dimension (8000 features) permet une séparation linéaire efficace des classes.

2. **Pas de surapprentissage**: La régularisation elasticnet (L1+L2) prévient le surapprentissage malgré la haute dimensionnalité.

3. **Rapidité**: Training en 4 minutes sur 10K échantillons.

4. **Scalabilité**: Peut gérer le dataset complet (84K) sans problème de mémoire.

### Limites

1. **Features images basiques**: Les histogrammes RGB ne capturent pas la sémantique visuelle (objets, formes, textures).

2. **Modèle linéaire**: Ne capture pas les interactions complexes entre features.

3. **TF-IDF statique**: Pas de compréhension contextuelle du texte.

## Optimisations appliquées

| Paramètre | Valeur initiale | Valeur optimale | Gain |
|-----------|----------------|-----------------|------|
| sample_size | 500 | 10000 | +27% |
| max_features | 5000 | 8000 | +4% |
| penalty | l2 | elasticnet | +2% |
| alpha | 0.0001 | 0.00005 | +0.4% |

## Comparaison avec d'autres approches

### vs Random Forest (50.8%)

Le SGDC surperforme car:
- L'espace TF-IDF haute dimension favorise la séparation linéaire
- Moins de paramètres à apprendre (meilleur ratio données/paramètres)
- Régularisation explicite plus efficace

### vs Transfer Learning (potentiel 85-90%)

Le Transfer Learning offrirait:
- Meilleurs features visuels (ResNet vs histogrammes)
- Features textuelles contextuelles (BERT vs TF-IDF)
- Fusion multimodale optimisée

Mais nécessite:
- 6-12h de training GPU
- Ressources computationnelles importantes

## Fichiers générés

### Métriques (models/SGDCModel/metrics/)

**metrics_summary.json**:
```json
{
  "accuracy": 0.754,
  "f1_weighted": 0.752,
  "precision_weighted": 0.758,
  "recall_weighted": 0.754
}
```

**classification_report.json**: Métriques détaillées par classe

**confusion_matrix.png**: Visualisation de la matrice de confusion

### Visualisations (models/SGDCModel/visualization/)

**feature_importance.png**: Top 30 features les plus importantes (valeur absolue des coefficients)

## Dépendances

- scikit-learn >= 1.0
- pandas >= 1.3
- numpy >= 1.21
- matplotlib >= 3.4
- joblib >= 1.0

Voir `requirements.txt` pour les versions exactes.

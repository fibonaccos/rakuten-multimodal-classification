# Preprocessing - Documentation

## Vue d'ensemble

Le preprocessing transforme les données brutes (texte + images) en features numériques exploitables par les modèles de machine learning.

## Architecture générale

Chaque modèle dispose de son propre module de preprocessing dans `src/preprocessing/[ModelName]/`:

```
src/preprocessing/[ModelName]/
├── preprocessing.yaml      # Configuration
├── config.py              # Chargement config
├── components.py          # Transformers individuels
├── pipeline.py            # Pipeline complet
├── __init__.py
└── __main__.py            # Point d'entrée
```

## Données d'entrée

### Fichiers requis

**Texte**:
- `X_train.csv` / `X_test.csv`: Colonnes `designation`, `description`
- Localisation configurable via `preprocessing.yaml`

**Images**:
- Dossier d'images (format JPG/PNG)
- Noms de fichiers correspondant aux IDs dans le CSV

**Labels**:
- `Y_train.csv` / `Y_test.csv`: Colonne `prdtypecode` (27 classes)

### Structure attendue

```csv
# X_train.csv
imageid,designation,description
123456,"Livre Python","Guide complet pour apprendre Python"
...

# Y_train.csv
imageid,prdtypecode
123456,10
...
```

## Pipeline de preprocessing

### 1. Chargement des données

```python
# Lecture CSV
X_train = pd.read_csv(text_train_path)
y_train = pd.read_csv(labels_train_path)

# Fusion
data = X_train.merge(y_train, on='imageid')
```

### 2. Sampling (optionnel)

Si `sample_size` < total ou `sample_size == -1`:

```python
if sample_size > 0:
    data = data.sample(n=sample_size, random_state=42)
else:
    data = data  # Tout le dataset
```

### 3. Feature extraction

#### Texte: TF-IDF

**Transformation**:
```python
TfidfVectorizer(
    max_features=8000,      # Nombre de features max
    ngram_range=(1, 2),     # Unigrammes + bigrammes
    min_df=2,               # Ignore mots très rares
    max_df=0.95,            # Ignore mots très fréquents
    lowercase=True,
    strip_accents='unicode'
)
```

**Étapes**:
1. Concatenation `designation + ' ' + description`
2. Tokenization (séparation en mots)
3. Calcul TF-IDF (Term Frequency - Inverse Document Frequency)
4. Normalisation L2

**Résultat**: Matrice sparse (n_samples, max_features)

#### Images: Histogrammes RGB

**Transformation**:
```python
def extract_color_histogram(image_path, bins=64):
    img = cv2.imread(image_path)
    hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256])
    return np.concatenate([hist_r, hist_g, hist_b]).flatten()
```

**Résultat**: Vecteur (bins * 3) par image (ex: 64*3 = 192 features)

### 4. Concatenation

```python
X_text = tfidf_matrix  # (n_samples, 8000)
X_images = image_features  # (n_samples, 192)

X_final = np.hstack([X_text, X_images])  # (n_samples, 8192)
```

### 5. Sauvegarde

```python
# Features + labels
train_df.to_csv('data/processed/[model]_train.csv', index=False)
test_df.to_csv('data/processed/[model]_test.csv', index=False)

# Transformers (pour réutilisation)
joblib.dump(tfidf_vectorizer, 'data/processed/[model]_tfidf.pkl')
```

## Configuration

### Paramètres communs

```yaml
preprocessing:
  config:
    sample_size: 10000      # -1 pour tout le dataset
    test_size: 0.2          # 20% pour test
    random_state: 42        # Reproductibilité
    
    logs:
      enable: true
      file_path: "logs/{DATE}_preprocessing.log"
```

### Paramètres texte

```yaml
  text:
    max_features: 8000      # Nombre max de features TF-IDF
    ngram_range: [1, 2]     # [min_n, max_n] pour n-grammes
    min_df: 2               # Min documents contenant le terme
    max_df: 0.95            # Max proportion de documents
    lowercase: true         # Convertir en minuscules
    strip_accents: unicode  # Retirer accents
```

**Impact des paramètres**:
- `max_features` ↑ → Plus de features, meilleur recall mais risque overfitting
- `ngram_range` [1, 3] → Ajoute trigrammes, capture expressions
- `min_df` ↑ → Ignore mots rares, réduit bruit
- `max_df` ↓ → Ignore mots trop fréquents (stop words)

### Paramètres images

```yaml
  images:
    hist_bins: 64           # Bins par canal RGB
    resize: [224, 224]      # Redimensionnement (optionnel)
    normalize: true         # Normalisation [0, 1]
```

**Impact**:
- `hist_bins` ↑ → Plus de détails couleur mais plus de features
- `resize` → Uniformise taille, accélère processing

## Différences entre modèles

### SGDC

```yaml
text:
  max_features: 8000        # Haute dimension OK
  ngram_range: [1, 2]       # Bigrammes utiles
```

**Raison**: SGDC gère bien haute dimensionnalité

### DecisionTree / Random Forest

```yaml
text:
  max_features: 3000        # Dimension réduite
  ngram_range: [1, 1]       # Unigrammes uniquement
```

**Raison**: Arbres sensibles à la malédiction de la dimensionnalité

## Utilisation

### Commande

```bash
python -m src.preprocessing.[ModelName]
```

**Exemples**:
```bash
python -m src.preprocessing.SGDCModel
python -m src.preprocessing.DecisionTreeModel
```

### Logs

Les logs sont sauvegardés dans `logs/`:

```
[2026-01-03 17:30:12] INFO: Starting preprocessing
[2026-01-03 17:30:15] INFO: Loaded 84000 samples
[2026-01-03 17:30:20] INFO: TF-IDF vectorization: 8000 features
[2026-01-03 17:32:45] INFO: Image processing: 192 features
[2026-01-03 17:32:50] INFO: Final dataset: (67200, 8192)
[2026-01-03 17:32:55] INFO: Saved to data/processed/
```

### Fichiers générés

```
data/processed/
├── [model]_train.csv       # Features + labels train
├── [model]_test.csv        # Features + labels test
├── [model]_tfidf.pkl       # TF-IDF vectorizer sauvegardé
└── [model]_label_enc.pkl   # Label encoder sauvegardé
```

## Optimisations possibles

### 1. Features textuelles enrichies

**Actuellement**: TF-IDF bag-of-words

**Améliorations**:
- Word embeddings (Word2Vec, GloVe)
- Sentence embeddings (BERT, DistilBERT)
- Features syntaxiques (POS tags)

**Impact attendu**: +5-10% accuracy

### 2. Features images avancées

**Actuellement**: Histogrammes RGB

**Améliorations**:
- HOG (Histogram of Oriented Gradients)
- SIFT/ORB keypoints
- CNN embeddings (ResNet, EfficientNet)

**Impact attendu**: +15-25% accuracy

### 3. Feature selection

**Méthode**:
```python
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=5000)
X_selected = selector.fit_transform(X, y)
```

**Avantage**: Réduit dimensionnalité, accélère training

### 4. Normalisation

**Actuellement**: TF-IDF normalisé L2, images [0, 1]

**Alternative**:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Impact**: Peut améliorer convergence pour certains modèles

## Gestion des données manquantes

### Texte manquant

```python
# Si designation ou description vide
data['designation'].fillna('', inplace=True)
data['description'].fillna('', inplace=True)
```

### Image manquante

```python
# Si image non trouvée, utiliser features nulles
if not os.path.exists(image_path):
    features = np.zeros(192)
else:
    features = extract_histogram(image_path)
```

## Reproductibilité

### Random seeds

```python
random_state = 42  # Fixé dans preprocessing.yaml

# Utilisé pour:
- train_test_split(random_state=42)
- data.sample(random_state=42)
- TfidfVectorizer(random_state=42)  # Si applicable
```

### Versions

Documenter les versions dans `requirements.txt`:
```
scikit-learn==1.3.0
pandas==1.5.3
opencv-python==4.8.0
```

## Performance

### Temps d'exécution

| Étape | SGDC (10K) | DecisionTree (5K) |
|-------|------------|-------------------|
| Chargement | 10s | 5s |
| TF-IDF | 30s | 15s |
| Images | 2min | 1min |
| Sauvegarde | 5s | 3s |
| **Total** | **~3min** | **~1.5min** |

### Utilisation mémoire

- TF-IDF sparse matrix: ~200 MB (10K samples, 8000 features)
- Images dense: ~15 MB (10K samples, 192 features)
- Pics lors du calcul: ~1 GB

**Recommandation**: Minimum 4 GB RAM

## Troubleshooting

### Erreur: Memory overflow

**Cause**: Trop de samples ou features

**Solution**:
```yaml
sample_size: 5000  # Réduire temporairement
max_features: 5000  # Réduire features
```

### Erreur: Images non trouvées

**Cause**: Chemin incorrect dans config

**Solution**: Vérifier `preprocessing.input.image_dir` dans YAML

### Erreur: Encoding CSV

**Cause**: Caractères spéciaux

**Solution**:
```python
pd.read_csv(path, encoding='utf-8')
# Ou
pd.read_csv(path, encoding='latin-1')
```

## Conclusion

Le preprocessing est une étape critique qui conditionne la performance des modèles. Un bon preprocessing inclut:

1. Features pertinentes (TF-IDF pour texte, histogrammes pour images)
2. Normalisation appropriée
3. Gestion des données manquantes
4. Reproductibilité (random seeds, versions)

Les optimisations futures devraient se concentrer sur l'amélioration des features images (CNN embeddings) pour gains significatifs.

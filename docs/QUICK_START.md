# Guide de Démarrage Rapide - Modèles Rakuten

Ce guide explique comment utiliser les modèles réorganisés pour le projet Rakuten.

## Structure du Projet

```
rakuten-multimodal-classification/
├── src/
│   ├── preprocessing/
│   │   ├── SGDCModel/           # Preprocessing pour SGDC
│   │   ├── DecisionTreeModel/   # Preprocessing pour Decision Tree
│   │   └── TLModel/             # Preprocessing pour Transfer Learning
│   └── models/
│       ├── SGDCModel/           # Modèle SGDC
│       ├── DecisionTreeModel/   # Modèle Decision Tree
│       └── TLModel/             # Modèle Transfer Learning
├── docs/
│   ├── preprocessing/           # Documentation preprocessing
│   ├── training/                # Documentation training
│   └── INTERPRETABILITY_GUIDE.md # Guide d'interprétabilité
├── data/
│   ├── raw/                     # Données brutes
│   └── clean/                   # Données prétraitées par modèle
└── models/                      # Modèles entraînés et artefacts
```

## Prérequis

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Dépendances principales
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- PyYAML
- colorlog
- Pillow (pour les images)
- torch, kornia (pour TLModel)
- tensorflow, keras (pour TLModel)

## Utilisation

### 1. SGDClassifier Model

#### Preprocessing
```bash
python -m src.preprocessing.SGDCModel
```

Configuration : `src/preprocessing/SGDCModel/preprocessing.yaml`
Documentation : `docs/preprocessing/SGDCModel.md`

#### Training
```bash
python -m src.models.SGDCModel --train
```

Configuration : `src/models/SGDCModel/model_config.yaml`
Documentation : `docs/training/SGDCModel.md`

#### Prediction
```bash
python -m src.models.SGDCModel --predict
```

#### Résultats
- Modèle : `models/SGDCModel/artefacts/sgdc_model.pkl`
- Métriques : `models/SGDCModel/metrics/`
- Visualisations : `models/SGDCModel/visualization/`

---

### 2. DecisionTreeClassifier Model

#### Preprocessing
```bash
python -m src.preprocessing.DecisionTreeModel
```

Configuration : `src/preprocessing/DecisionTreeModel/preprocessing.yaml`
Documentation : `docs/preprocessing/DecisionTreeModel.md`

#### Training
```bash
python -m src.models.DecisionTreeModel --train
```

Configuration : `src/models/DecisionTreeModel/model_config.yaml`
Documentation : `docs/training/DecisionTreeModel.md`

#### Prediction
```bash
python -m src.models.DecisionTreeModel --predict
```

#### Résultats
- Modèle : `models/DecisionTreeModel/artefacts/decision_tree_model.pkl`
- Métriques : `models/DecisionTreeModel/metrics/`
- Structure de l'arbre : `models/DecisionTreeModel/artefacts/tree_structure.txt`
- Visualisations : `models/DecisionTreeModel/visualization/`

---

### 3. Transfer Learning Model (ResNet)

#### Preprocessing
```bash
python -m src.preprocessing.TLModel
```

Configuration : `src/preprocessing/TLModel/preprocessing.yaml`
Documentation : `docs/preprocessing/TLModel.md`

#### Training
```bash
python -m src.models.TLModel --train
```

Configuration : `src/models/TLModel/model_config.yaml`
Documentation : `docs/training/TLModel.md`

#### Prediction
```bash
python -m src.models.TLModel --predict
```

#### Résultats
- Modèle : `models/TLModel/artefacts/best_model.keras`
- Métriques : `models/TLModel/metrics/`
- Historiques : `models/TLModel/artefacts/fit_history.pkl`
- Visualisations : `models/TLModel/records/`

---

## Workflow Complet

### Pour SGDC ou DecisionTree

```bash
# 1. Preprocessing
python -m src.preprocessing.SGDCModel  # ou DecisionTreeModel

# 2. Training
python -m src.models.SGDCModel --train  # ou DecisionTreeModel

# 3. Analyser les résultats
# Voir docs/INTERPRETABILITY_GUIDE.md

# 4. Prediction
python -m src.models.SGDCModel --predict  # ou DecisionTreeModel
```

### Pour Transfer Learning

```bash
# 1. Preprocessing (peut être long)
python -m src.preprocessing.TLModel

# 2. Training (très long, GPU recommandé)
python -m src.models.TLModel --train

# 3. Analyser les résultats
# Voir docs/INTERPRETABILITY_GUIDE.md

# 4. Prediction
python -m src.models.TLModel --predict
```

---

## Configuration

Chaque modèle a deux fichiers de configuration YAML :

### 1. Preprocessing (`src/preprocessing/[MODEL]/preprocessing.yaml`)

Paramètres clés :
- `sample_size` : Nombre d'exemples (-1 pour tout)
- `train_size` : Ratio train/test (0.8 par défaut)
- `random_state` : Seed pour reproductibilité
- `steps` : Active/désactive les étapes de preprocessing

### 2. Training (`src/models/[MODEL]/model_config.yaml`)

Paramètres clés :
- **SGDC** : `loss`, `penalty`, `alpha`, `epochs`
- **DecisionTree** : `max_depth`, `min_samples_split`, `criterion`
- **TLModel** : `epochs`, `batch_size`, `learning_rate`, `optimizer`

---

## Debugging

### Problème : Données non trouvées
```
FileNotFoundError: data/raw/X_raw.csv
```

**Solution** : Vérifier les chemins dans le fichier `preprocessing.yaml`

### Problème : Mémoire insuffisante (TLModel)
```
MemoryError ou CUDA out of memory
```

**Solution** :
- Réduire `batch_size` dans `model_config.yaml`
- Réduire `n_images` dans `preprocessing.yaml`
- Désactiver CUDA si pas assez de VRAM

### Problème : Surapprentissage (DecisionTree)
```
overfitting_gap > 0.15
```

**Solution** : Ajuster dans `model_config.yaml` :
- Réduire `max_depth` (ex: 10)
- Augmenter `min_samples_split` (ex: 20)
- Utiliser `ccp_alpha` (ex: 0.001)

### Problème : Performance faible
```
accuracy < 0.60
```

**Solutions** :
1. Augmenter `sample_size` (plus de données)
2. Ajuster les paramètres du modèle
3. Vérifier la qualité des features
4. Voir `docs/INTERPRETABILITY_GUIDE.md`

---

## Logs

Les logs sont sauvegardés dans `.logs/` :
- Preprocessing : `.logs/preprocessing/`
- Training : `.logs/models/`

Format : `{DATE}_{MODEL}_{JOB}.log`

---

## Intégration avec Streamlit

Pour intégrer un modèle dans Streamlit :

1. **Charger le modèle**
```python
import pickle

with open('models/SGDCModel/artefacts/sgdc_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/SGDCModel/artefacts/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
```

2. **Prétraiter les données**
```python
from src.preprocessing.SGDCModel import TextCleaner, TextVectorizer

# Utiliser les transformateurs sauvegardés
with open('data/clean/sgdc_model/transformers.pkl', 'rb') as f:
    transformers = pickle.load(f)
```

3. **Faire une prédiction**
```python
features = preprocess_input(input_data)
prediction = model.predict(features)
label = label_encoder.inverse_transform(prediction)
```

---

## Prochaines Étapes

1. **Tester les modèles** sur vos données
2. **Analyser les résultats** avec le guide d'interprétabilité
3. **Ajuster les hyperparamètres** si nécessaire
4. **Comparer les modèles** pour choisir le meilleur
5. **Intégrer dans Streamlit** pour la démo

---

## Support

- Documentation preprocessing : `docs/preprocessing/[MODEL].md`
- Documentation training : `docs/training/[MODEL].md`
- Guide d'interprétabilité : `docs/INTERPRETABILITY_GUIDE.md`

Pour plus d'aide, consulter les fichiers de configuration YAML qui contiennent des commentaires détaillés.

# Rakuten Multimodal Classification

Projet de classification multimodale pour le challenge Rakuten, utilisant des données textuelles et images.

## 🚀 Démarrage Rapide

**Guides principaux** :
- 📖 [Guide de démarrage rapide](docs/QUICK_START.md)
- 🔍 [Guide d'interprétabilité](docs/INTERPRETABILITY_GUIDE.md)

## 🎯 Modèles Disponibles

Ce projet propose **trois modèles** organisés de manière uniforme :

### 1. SGDClassifier
- **Type** : Modèle linéaire avec descente de gradient stochastique
- **Forces** : Rapide, scalable, bonne performance sur texte
- **Usage** : Classification avec features TF-IDF + histogrammes couleur
- 📁 Code : `src/models/SGDCModel/`
- 📚 Docs : [Preprocessing](docs/preprocessing/SGDCModel.md) | [Training](docs/training/SGDCModel.md)

### 2. DecisionTreeClassifier
- **Type** : Arbre de décision
- **Forces** : Interprétable, règles explicites
- **Usage** : Analyse des features importantes, baseline
- 📁 Code : `src/models/DecisionTreeModel/`
- 📚 Docs : [Preprocessing](docs/preprocessing/DecisionTreeModel.md) | [Training](docs/training/DecisionTreeModel.md)

### 3. Transfer Learning (ResNet)
- **Type** : CNN pré-entraîné avec fine-tuning
- **Forces** : Excellente performance sur images
- **Usage** : Classification basée principalement sur les images
- 📁 Code : `src/models/TLModel/`
- 📚 Docs : [Preprocessing](docs/preprocessing/TLModel.md) | [Training](docs/training/TLModel.md)

## 📂 Organisation du Projet

```
rakuten-multimodal-classification/
├── src/
│   ├── preprocessing/          # Pipelines de preprocessing par modèle
│   │   ├── SGDCModel/
│   │   ├── DecisionTreeModel/
│   │   └── TLModel/
│   └── models/                 # Code des modèles
│       ├── SGDCModel/
│       ├── DecisionTreeModel/
│       └── TLModel/
├── docs/
│   ├── preprocessing/          # Documentation preprocessing
│   ├── training/               # Documentation training
│   ├── QUICK_START.md         # Guide de démarrage
│   └── INTERPRETABILITY_GUIDE.md  # Guide d'interprétabilité
├── data/
│   ├── raw/                    # Données brutes
│   └── clean/                  # Données prétraitées par modèle
├── models/                     # Modèles entraînés et artefacts
│   ├── SGDCModel/
│   ├── DecisionTreeModel/
│   └── TLModel/
├── notebooks/                  # Explorations et analyses
├── reports/                    # Rapports de projet
└── requirements.txt            # Dépendances Python
```

## 🛠️ Installation

```bash
# Cloner le repository
git clone https://github.com/votre-repo/rakuten-multimodal-classification.git
cd rakuten-multimodal-classification

# Installer les dépendances
pip install -r requirements.txt
```

## 📝 Utilisation Rapide

### Exemple avec SGDClassifier

```bash
# 1. Preprocessing
python -m src.preprocessing.SGDCModel

# 2. Training
python -m src.models.SGDCModel --train

# 3. Prediction
python -m src.models.SGDCModel --predict
```

### Exemple avec DecisionTreeClassifier

```bash
# 1. Preprocessing
python -m src.preprocessing.DecisionTreeModel

# 2. Training
python -m src.models.DecisionTreeModel --train

# 3. Prediction
python -m src.models.DecisionTreeModel --predict
```

Voir [QUICK_START.md](docs/QUICK_START.md) pour plus de détails.

## ⚙️ Configuration

Chaque modèle dispose de **deux fichiers YAML** :

1. **Preprocessing** : `src/preprocessing/[MODEL]/preprocessing.yaml`
   - Chemins des données
   - Paramètres de prétraitement
   - Features à extraire

2. **Training** : `src/models/[MODEL]/model_config.yaml`
   - Hyperparamètres du modèle
   - Chemins des artefacts
   - Métriques à calculer

## 📊 Résultats et Interprétabilité

Chaque modèle génère :
- ✅ **Métriques** : Accuracy, F1-score, Precision, Recall
- 📈 **Visualisations** : Matrice de confusion, importance des features
- 💾 **Artefacts** : Modèle entraîné, encodeurs, historiques

Voir le [Guide d'Interprétabilité](docs/INTERPRETABILITY_GUIDE.md) pour analyser les résultats.

## 🏗️ Architecture Unifiée

Tous les modèles suivent la **même structure** inspirée de TLModel :

```
[ModelName]/
├── __init__.py
├── __main__.py           # Point d'entrée (--train, --predict)
├── config.py             # Chargement config YAML
├── components.py         # Transformateurs preprocessing (si applicable)
├── pipeline.py           # Pipeline preprocessing (si applicable)
├── model.py              # Définition modèle
├── train.py              # Script d'entraînement
├── predict.py            # Script de prédiction
└── [model]_config.yaml   # Configuration
```

**Avantages** :
- ✨ Cohérence entre les modèles
- 🔄 Reproductibilité garantie
- 📦 Facile à intégrer dans Streamlit
- 🛠️ Maintenance simplifiée

## 🔬 Workflow de Développement

### 1. Preprocessing
Chaque modèle a son propre preprocessing adapté :
```bash
python -m src.preprocessing.[ModelName]
```

### 2. Training
Entraînement avec logging et sauvegarde automatique :
```bash
python -m src.models.[ModelName] --train
```

### 3. Analyse des Résultats
Consulter :
- `models/[ModelName]/metrics/` : Métriques quantitatives
- `models/[ModelName]/visualization/` : Graphiques
- Le [Guide d'Interprétabilité](docs/INTERPRETABILITY_GUIDE.md)

### 4. Ajustement
Modifier les fichiers YAML de configuration et ré-entraîner.

### 5. Prédiction
```bash
python -m src.models.[ModelName] --predict
```

## 📈 Comparaison des Modèles

| Critère | SGDC | DecisionTree | Transfer Learning |
|---------|------|--------------|-------------------|
| **Performance texte** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Performance image** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Vitesse** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Interprétabilité** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Mémoire requise** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Risque overfitting** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🤝 Contribution

Pour ajouter un nouveau modèle :
1. Créer `src/preprocessing/[NewModel]/` avec la structure standard
2. Créer `src/models/[NewModel]/` avec la structure standard
3. Ajouter la documentation dans `docs/`
4. Suivre les conventions de nommage et structure

## 📄 Licence

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.

## 👥 Équipe

Projet réalisé dans le cadre de la formation DataScientest.

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>

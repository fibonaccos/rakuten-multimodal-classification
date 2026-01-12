# 📁 Data Directory - Modèles et Transformers

Ce dossier contient les modèles entraînés et les transformers nécessaires pour la démonstration Streamlit.

## 📂 Structure

```
data/
├── models/
│   ├── SGDCModel/
│   │   ├── sgdc_model.pkl          (3.3 MB)
│   │   └── label_encoder.pkl       (0.45 KB)
│   └── RandomForest/
│       ├── rf_model.pkl            (1.9 MB)
│       └── label_encoder.pkl       (0.45 KB)
└── transformers/
    └── transformers_sgdc.pkl       (904 KB)
```

## 🚀 Configuration

### Les fichiers sont déjà présents

Les modèles et transformers sont déjà copiés dans ce dossier et prêts à l'emploi.

### Si les fichiers sont manquants

Si vous avez cloné le repo et que les fichiers .pkl ne sont pas présents (ignorés par git), vous pouvez :

1. **Copier depuis Rakuten_Streamlit_Presentation** (si vous l'avez) :
   ```bash
   # Depuis la racine du projet
   mkdir -p src/streamlit/app/data/models/SGDCModel
   mkdir -p src/streamlit/app/data/models/RandomForest
   mkdir -p src/streamlit/app/data/transformers
   
   # Copier SGDC
   cp Rakuten_Streamlit_Presentation/models/SGDCModel/artefacts/sgdc_model.pkl src/streamlit/app/data/models/SGDCModel/
   cp Rakuten_Streamlit_Presentation/models/SGDCModel/artefacts/label_encoder.pkl src/streamlit/app/data/models/SGDCModel/
   
   # Copier Random Forest
   cp Rakuten_Streamlit_Presentation/models/RandomForest/artefacts/rf_model.pkl src/streamlit/app/data/models/RandomForest/
   cp Rakuten_Streamlit_Presentation/models/RandomForest/artefacts/label_encoder.pkl src/streamlit/app/data/models/RandomForest/
   
   # Copier transformers
   cp Rakuten_Streamlit_Presentation/data/clean/sgdc_model/transformers.pkl src/streamlit/app/data/transformers/transformers_sgdc.pkl
   ```

2. **Ou réentraîner les modèles** :
   ```bash
   # Se placer sur la branche avec le code d'entraînement
   git checkout reorg_sgdc_classif
   
   # Lancer preprocessing + training
   python -m src.preprocessing.SGDCModel
   python -m src.models.SGDCModel
   
   # Copier les artefacts générés
   # ...
   ```

## ⚠️ Important

**Ces fichiers ne sont PAS versionnés dans git** (trop lourds, binaires).

Chaque membre de l'équipe doit :
- Soit avoir accès à `Rakuten_Streamlit_Presentation/`
- Soit copier les fichiers depuis un collègue
- Soit réentraîner les modèles

## 🔍 Vérification

Pour vérifier que les fichiers sont bien présents :

```bash
ls -lh src/streamlit/app/data/models/SGDCModel/
ls -lh src/streamlit/app/data/models/RandomForest/
ls -lh src/streamlit/app/data/transformers/
```

Vous devriez voir les fichiers .pkl listés.

## 📝 Note

Le fichier `.gitignore` dans ce dossier empêche le commit des fichiers .pkl pour éviter de polluer le repository avec des fichiers binaires lourds.

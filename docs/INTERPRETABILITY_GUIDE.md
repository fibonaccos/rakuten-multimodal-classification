# Guide d'Interprétabilité des Modèles

Ce guide explique comment interpréter les résultats de chaque modèle du projet Rakuten.

## Table des matières

1. [SGDClassifier](#sgdclassifier)
2. [DecisionTreeClassifier](#decisiontreeclassifier)
3. [Transfer Learning Model (ResNet)](#transfer-learning-model)

---

## SGDClassifier

### Localisation des résultats
- **Métriques** : `models/SGDCModel/metrics/`
- **Visualisations** : `models/SGDCModel/visualization/`
- **Artefacts** : `models/SGDCModel/artefacts/`

### Métriques principales

#### 1. Accuracy et F1-Score
Fichier : `metrics_summary.json`

```json
{
  "accuracy": 0.75,
  "f1_macro": 0.72,
  "f1_weighted": 0.74
}
```

- **Accuracy** : Pourcentage de prédictions correctes (global)
- **F1 (macro)** : Moyenne du F1 pour chaque classe (toutes égales)
- **F1 (weighted)** : Moyenne du F1 pondérée par le nombre d'exemples par classe

**Interprétation** :
- Accuracy > 0.70 : Bon modèle
- F1 macro << F1 weighted : Classes déséquilibrées, certaines classes mal prédites
- F1 macro ≈ F1 weighted : Performance uniforme sur toutes les classes

#### 2. Rapport de classification
Fichier : `classification_report.json`

Pour chaque classe :
- **precision** : Parmi les prédictions de cette classe, combien sont correctes
- **recall** : Parmi les vrais exemples de cette classe, combien sont détectés
- **f1-score** : Moyenne harmonique de precision et recall

**Classes à surveiller** :
- Precision faible : Le modèle confond cette classe avec d'autres
- Recall faible : Le modèle rate beaucoup d'exemples de cette classe
- Support faible : Peu d'exemples dans les données

#### 3. Matrice de confusion
Fichier : `confusion_matrix.png`

**Comment lire** :
- Diagonale : Prédictions correctes
- Hors diagonale : Confusions entre classes
- Ligne i, colonne j : Le modèle a prédit j alors que c'était i

**Analyse** :
- Chercher les confusions fréquentes (valeurs élevées hors diagonale)
- Identifier les paires de classes problématiques
- Comparer avec les classes visuellement similaires

#### 4. Importance des features
Fichier : `feature_importance.png`

**Interprétation** :
- Coefficients positifs/négatifs : Impact sur la probabilité de la classe
- Valeur absolue élevée : Feature très influente
- Features textuelles (TF-IDF) : Mots/bigrammes importants
- Features d'images : Canaux de couleur dominants

**Utilisation** :
- Identifier les mots-clés discriminants
- Vérifier la cohérence avec le domaine métier
- Détecter les features inutiles ou bruitées

---

## DecisionTreeClassifier

### Localisation des résultats
- **Métriques** : `models/DecisionTreeModel/metrics/`
- **Visualisations** : `models/DecisionTreeModel/visualization/`
- **Artefacts** : `models/DecisionTreeModel/artefacts/`

### Métriques principales

#### 1. Accuracy et détection du surapprentissage
Fichier : `metrics_summary.json`

```json
{
  "accuracy": 0.68,
  "train_accuracy": 0.95,
  "overfitting_gap": 0.27,
  "tree_depth": 45,
  "n_leaves": 1234
}
```

**Interprétation** :
- **overfitting_gap > 0.15** : Surapprentissage détecté !
- **tree_depth > 20** : Arbre trop profond, risque de surapprentissage
- **n_leaves > 500** : Arbre trop complexe

**Actions correctives** :
1. Réduire `max_depth` (ex: 10-15)
2. Augmenter `min_samples_split` (ex: 20-50)
3. Utiliser `ccp_alpha` pour l'élagage (ex: 0.001)

#### 2. Structure de l'arbre
Fichier : `tree_structure.txt`

**Exemple** :
```
|--- feature_12 <= 0.35
|   |--- feature_45 <= 0.12
|   |   |--- class: 1040
|   |--- feature_45 >  0.12
|   |   |--- class: 2280
```

**Interprétation** :
- Identifier les règles de décision principales
- Vérifier la cohérence des seuils
- Comprendre le raisonnement du modèle

#### 3. Visualisation de l'arbre
Fichier : `tree_visualization.png` (si arbre < 100 feuilles)

**Comment lire** :
- Couleur des nœuds : Classe majoritaire
- Intensité de couleur : Pureté du nœud
- Profondeur : Complexité des règles

#### 4. Importance des features
Fichier : `feature_importance.png`

**Interprétation** :
- Basée sur l'impureté de Gini
- Mesure l'utilité pour diviser les données
- Valeur élevée : Feature utilisée tôt dans l'arbre

**Différence avec SGDC** :
- SGDC : Impact linéaire sur la décision
- DecisionTree : Utilité pour la séparation des classes

---

## Transfer Learning Model

### Localisation des résultats
- **Métriques** : `models/TLModel/metrics/`
- **Visualisations** : `models/TLModel/visualization/`
- **Artefacts** : `models/TLModel/artefacts/`

### Métriques principales

#### 1. Courbes d'apprentissage
Fichiers : `fit_plots.jpg`, `validation_plots.jpg`

**Graphiques** :
- **Loss** : Erreur du modèle (plus bas = mieux)
- **Accuracy** : Précision (plus haut = mieux)

**Interprétation** :
- Loss train décroît, loss val stable : Bon apprentissage
- Loss train décroît, loss val augmente : Surapprentissage
- Loss train et val stagnent : Modèle ne progresse plus
- Oscillations importantes : Learning rate trop élevé

**Actions** :
- Surapprentissage : Réduire epochs, augmenter dropout, augmenter augmentation
- Stagnation : Ajuster learning rate, changer optimizer
- Oscillations : Réduire learning rate

#### 2. Métriques par classe
Fichier : Dossier `class_validation_plots_dir/`

**Pour chaque classe** :
- Precision, Recall, F1-score au cours de l'entraînement
- Identifier les classes difficiles à apprendre

#### 3. Matrice de confusion
Fichier : `test_confusion_matrix.png`

Même interprétation que pour SGDC.

---

## Comparaison des modèles

| Critère | SGDC | DecisionTree | Transfer Learning |
|---------|------|--------------|-------------------|
| **Interprétabilité** | Moyenne (coefficients) | Excellente (règles) | Faible (boîte noire) |
| **Overfitting** | Contrôlé par régularisation | Risque élevé | Risque moyen |
| **Performance texte** | Excellente | Bonne | Moyenne |
| **Performance image** | Moyenne | Moyenne | Excellente |
| **Temps d'entraînement** | Rapide | Rapide | Lent |
| **Mémoire requise** | Faible | Moyenne | Élevée |

---

## Workflow d'analyse des résultats

### 1. Vérifier les métriques globales
```bash
# Voir le résumé
cat models/[MODEL]/metrics/metrics_summary.json
```

### 2. Analyser la matrice de confusion
- Identifier les confusions principales
- Chercher les patterns (classes similaires confondues)

### 3. Examiner l'importance des features
- Vérifier la cohérence avec le domaine
- Identifier les features bruitées

### 4. Vérifier le surapprentissage
- Comparer train vs test accuracy
- Analyser les courbes d'apprentissage (TL)
- Vérifier tree_depth et n_leaves (DT)

### 5. Analyser par classe
- Identifier les classes problématiques
- Vérifier le support (nombre d'exemples)
- Comparer precision vs recall

### 6. Décisions d'amélioration

**Si accuracy faible** :
- Augmenter les features
- Essayer d'autres modèles
- Collecter plus de données

**Si surapprentissage** :
- Régularisation (alpha pour SGDC/DT)
- Dropout (TL)
- Réduire la complexité (max_depth)
- Augmentation de données

**Si classes déséquilibrées** :
- Équilibrer les données (SMOTE, over/under-sampling)
- Ajuster class_weight
- Utiliser F1-score au lieu d'accuracy

**Si confusions entre classes** :
- Analyser les caractéristiques des classes confondues
- Ajouter des features discriminantes
- Utiliser des features spécifiques

---

## Checklist finale avant production

- [ ] Accuracy > 0.70 sur le test
- [ ] Overfitting gap < 0.15
- [ ] F1-score uniforme sur toutes les classes (ou justifié)
- [ ] Confusion matrix cohérente
- [ ] Features importantes cohérentes avec le domaine
- [ ] Documentation complète des résultats
- [ ] Modèle et artefacts sauvegardés
- [ ] Tests de prédiction validés

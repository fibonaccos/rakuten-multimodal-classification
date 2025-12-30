# Entraînement du modèle DecisionTreeClassifier

L'entraînement du modèle se fait par l'intermédiaire du fichier de configuration disponible et éditable [ici](../../src/models/DecisionTreeModel/model_config.yaml). La section concernée est celle dont la clé principale est `train`.

Il est nécessaire d'avoir exécuté en amont le pipeline de preprocessing pour s'assurer entre autres :

- d'avoir les features préparées et combinées,
- d'assurer une reproductibilité de l'entraînement,
- d'assurer une performance significative du modèle entraîné.

Enfin, la phase d'entraînement intègre la phase de test qui permet la mesure des performances suivant différentes métriques.

## Configuration

La configuration est exposée sous format **yaml**. On y trouvera les éléments suivants au sein de la clé `train` :

- **config** : les paramètres généraux spécifiques à l'entraînement.
- **data_dir** : les chemins d'accès aux données d'entraînement et de test.
- **artefacts** : les différents objets générés à l'issue de l'entraînement.
- **metrics** : les différentes métriques mesurées et sauvegardées à l'issue de la phase de test.
- **visualization** : les différents résultats graphiques générés.

### Paramètres du modèle

Le modèle DecisionTreeClassifier est configuré avec les paramètres suivants :

- **criterion** : critère de division (`gini` ou `entropy`)
- **max_depth** : profondeur maximale de l'arbre (`null` pour illimité)
- **min_samples_split** : nombre minimum d'échantillons pour diviser un nœud
- **min_samples_leaf** : nombre minimum d'échantillons dans une feuille
- **max_features** : nombre maximum de features à considérer (`null` pour toutes)
- **max_leaf_nodes** : nombre maximum de feuilles (`null` pour illimité)
- **ccp_alpha** : paramètre de complexité pour l'élagage

### Chemins

L'entraînement nécessite l'accès à différents fichiers et répertoires (pré-existants ou non) :

- **train.config.logs.file_path** : chemin d'accès au fichier de log, *créé si besoin*.
- **train.data_dir.train_features** : chemin d'accès aux features d'entraînement.
- **train.data_dir.test_features** : chemin d'accès aux features de test.
- **train.data_dir.train_labels** : chemin d'accès aux labels d'entraînement.
- **train.data_dir.test_labels** : chemin d'accès aux labels de test.
- **train.artefacts.base_dir** : chemin d'accès au dossier contenant les artefacts, *créé si besoin*.
- **train.artefacts.model_file** : chemin d'accès au modèle entraîné, *créé si besoin*.
- **train.artefacts.label_encoder** : chemin d'accès au label encoder, *créé si besoin*.
- **train.artefacts.tree_structure** : chemin d'accès à la structure textuelle de l'arbre, *créé si besoin*.
- **train.metrics.base_dir** : chemin d'accès au dossier contenant les métriques, *créé si besoin*.
- **train.metrics.confusion_matrix** : chemin d'accès à la matrice de confusion, *créé si besoin*.
- **train.metrics.classification_report** : chemin d'accès au rapport de classification, *créé si besoin*.
- **train.metrics.metrics_summary** : chemin d'accès au résumé des métriques, *créé si besoin*.
- **train.visualization.base_dir** : chemin d'accès au dossier de visualisation, *créé si besoin*.
- **train.visualization.tree_visualization** : chemin d'accès à la visualisation de l'arbre, *créé si besoin*.
- **train.visualization.feature_importance** : chemin d'accès au graphique d'importance des features, *créé si besoin*.

## Exécution de l'entraînement

Pour lancer l'entraînement, utilisez la commande suivante depuis la racine du projet :

```bash
python -m src.models.DecisionTreeModel --train
```

L'entraînement génèrera :
- Le modèle entraîné sauvegardé (format pickle)
- Le label encoder pour convertir les prédictions
- Les métriques de performance (accuracy, F1, precision, recall)
- La matrice de confusion
- Un rapport de classification détaillé par classe
- Un graphique d'importance des features
- Une visualisation de l'arbre (si suffisamment petit)
- La structure textuelle de l'arbre

## Métriques générées

Le modèle est évalué sur plusieurs métriques :

- **Accuracy** : précision globale sur le test
- **Train Accuracy** : précision sur l'entraînement (pour détecter le surapprentissage)
- **Overfitting Gap** : différence entre train et test accuracy
- **F1 Score (macro)** : moyenne non pondérée du F1 par classe
- **F1 Score (weighted)** : moyenne pondérée du F1 par classe
- **Precision (weighted)** : précision moyenne pondérée
- **Recall (weighted)** : rappel moyen pondéré
- **Tree Depth** : profondeur de l'arbre entraîné
- **Number of Leaves** : nombre de feuilles

Un rapport de classification détaillé est également généré pour chaque classe avec les métriques individuelles.

## Détection du surapprentissage

Le système détecte automatiquement le surapprentissage en comparant l'accuracy sur l'entraînement et le test :
- **Gap < 0.15** : Modèle sain
- **Gap > 0.15** : Surapprentissage potentiel, ajuster les hyperparamètres

Pour réduire le surapprentissage :
1. Réduire `max_depth` (ex: 10-20)
2. Augmenter `min_samples_split` (ex: 10-50)
3. Augmenter `min_samples_leaf` (ex: 5-20)
4. Utiliser `ccp_alpha` pour l'élagage (ex: 0.001-0.01)

## Interprétabilité

### Importance des features

Le modèle DecisionTree permet d'analyser l'importance des features via l'impureté de Gini. Un graphique des 20 features les plus importantes est généré automatiquement.

L'importance d'une feature indique :
- À quel point elle est utilisée pour diviser les données
- Son impact sur la réduction de l'impureté
- Valeur élevée = feature très influente dans la décision

### Structure de l'arbre

La structure textuelle de l'arbre est exportée dans un fichier `.txt` pour une analyse détaillée des règles de décision. Cela permet de :
- Comprendre les règles de décision exactes
- Identifier les seuils de décision
- Vérifier la logique du modèle

### Visualisation de l'arbre

Pour les arbres de petite taille (< 100 feuilles), une visualisation graphique est générée. La visualisation est limitée à une profondeur de 3 niveaux pour la lisibilité.

### Matrice de confusion

La matrice de confusion permet de visualiser :
- Les prédictions correctes (diagonale)
- Les confusions entre classes (hors diagonale)
- Les classes problématiques

## Prédiction

Pour faire des prédictions avec le modèle entraîné :

```bash
python -m src.models.DecisionTreeModel --predict
```

Les prédictions sont sauvegardées dans le dossier configuré, avec les probabilités par classe.

## Recommandations

1. **Commencer avec des paramètres conservateurs** : max_depth=10, min_samples_split=20
2. **Surveiller le surapprentissage** : Vérifier l'écart train/test accuracy
3. **Analyser l'importance des features** : Identifier les features clés
4. **Étudier la structure de l'arbre** : Vérifier la cohérence des règles
5. **Comparer avec d'autres modèles** : Les arbres de décision peuvent être surpassés par des ensembles (Random Forest, XGBoost)

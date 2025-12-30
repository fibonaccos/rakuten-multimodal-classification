# Entraînement du modèle SGDClassifier

L'entraînement du modèle se fait par l'intermédiaire du fichier de configuration disponible et éditable [ici](../../src/models/SGDCModel/model_config.yaml). La section concernée est celle dont la clé principale est `train`.

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

Le modèle SGDClassifier est configuré avec les paramètres suivants :

- **loss** : fonction de perte (par défaut : `log_loss` pour la régression logistique)
- **penalty** : type de régularisation (`l1`, `l2`, ou `elasticnet`)
- **alpha** : force de régularisation
- **epochs** : nombre maximum d'itérations
- **learning_rate** : stratégie de taux d'apprentissage (`optimal`, `constant`, `adaptive`)
- **early_stopping** : arrêt anticipé si pas d'amélioration
- **validation_fraction** : fraction des données pour la validation
- **n_iter_no_change** : nombre d'itérations sans amélioration avant arrêt

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
- **train.metrics.base_dir** : chemin d'accès au dossier contenant les métriques, *créé si besoin*.
- **train.metrics.confusion_matrix** : chemin d'accès à la matrice de confusion, *créé si besoin*.
- **train.metrics.classification_report** : chemin d'accès au rapport de classification, *créé si besoin*.
- **train.metrics.metrics_summary** : chemin d'accès au résumé des métriques, *créé si besoin*.
- **train.visualization.base_dir** : chemin d'accès au dossier de visualisation, *créé si besoin*.
- **train.visualization.feature_importance** : chemin d'accès au graphique d'importance des features, *créé si besoin*.

## Exécution de l'entraînement

Pour lancer l'entraînement, utilisez la commande suivante depuis la racine du projet :

```bash
python -m src.models.SGDCModel --train
```

L'entraînement génèrera :
- Le modèle entraîné sauvegardé (format pickle)
- Le label encoder pour convertir les prédictions
- Les métriques de performance (accuracy, F1, precision, recall)
- La matrice de confusion
- Un rapport de classification détaillé par classe
- Un graphique d'importance des features

## Métriques générées

Le modèle est évalué sur plusieurs métriques :

- **Accuracy** : précision globale
- **F1 Score (macro)** : moyenne non pondérée du F1 par classe
- **F1 Score (weighted)** : moyenne pondérée du F1 par classe
- **Precision (weighted)** : précision moyenne pondérée
- **Recall (weighted)** : rappel moyen pondéré

Un rapport de classification détaillé est également généré pour chaque classe avec les métriques individuelles.

## Interprétabilité

### Importance des features

Le modèle SGDClassifier permet d'analyser l'importance des features via les coefficients du modèle. Un graphique des 20 features les plus importantes est généré automatiquement.

Les coefficients représentent l'impact de chaque feature sur la décision du modèle :
- Valeurs positives : augmentent la probabilité de la classe
- Valeurs négatives : diminuent la probabilité de la classe
- Valeur absolue élevée : feature très influente

### Matrice de confusion

La matrice de confusion permet de visualiser :
- Les prédictions correctes (diagonale)
- Les confusions entre classes (hors diagonale)
- Les classes problématiques

## Prédiction

Pour faire des prédictions avec le modèle entraîné :

```bash
python -m src.models.SGDCModel --predict
```

Les prédictions sont sauvegardées dans le dossier configuré, avec les probabilités par classe si disponibles.

# Entraînement du modèle de transfer learning

L'entraînement du modèle se fait par l'intermédiaire du fichier de configuration disponible et éditable [ici](../../src/models/TLModel/model_config.yaml). La section concernée est celle dont la clé principale est `train`.

Il est nécessaire d'avoir exécuté en amont le pipeline de preprocessing pour s'assurer entre autres :

- d'avoir la bonne structure du dataset utilisé pour l'entraînement,
- d'assurer une reproductibilité de l'entraînement,
- d'assurer une performance significative du modèle entraîné.

Enfin, la phase d'entraînement peut au choix intégrer la phase de test qui permet la mesure des performances suivants différentes métriques.

## Configuration

La configuration est exposée sous format **yaml**. On y trouvera les éléments suivants au sein de la clé `train` :

- **config** : les paramètres généraux spécifiques à l'entraînement.
- **data_dir** : les chemins d'accès aux datasets d'entraînement et de test.
- **artefacts** : les différents objets générés à l'issue de l'entraînement, hors métriques.
- **metrics** : les différentes métriques mesurées et sauvegardées à l'issue de la phase de test.
- **records** : les différents résultats graphiques de métriques réalisées pendant l'entraînement et le test.

### Chemins

L'entraînement nécessite l'accès à différents fichiers et répertoires (pré-existants ou non) :

- **train.config.logs.file_path** : chemin d'acès au fichier de log, *créé si besoin*. Il est conseillé de conserver le préfixe `{DATE}_` pour le nom du fichier.
- **train.data_dir.train** : chemin d'accès au dataset d'entraînement.
- **train.data_dir.test** : chemin d'accès au dataset de test.
- **train.artefacts.base_dir** : chemin d'accès au dossier contenant les artefacts, *créé si besoin*.
- **train.artefacts.labels_name** : chemin d'accès au fichier **.json** contenant le nom des labels, *créé si besoin*.
- **train.artefacts.best_model** : chemin d'accès au fichier **.keras** contenant le meilleur modèle retenu, *créé si besoin*.
- **train.artefacts.fit_history** : chemin d'accès au fichier **.pkl** contenant l'historique des quantités `train.config.loss` et `train.config.metric` lors de l'entraînement, *créé si besoin*.
- **train.artefacts.callback_history** : chemin d'accès au fichier **.pkl** contenant l'historique des métriques additionnelles lors de l'entraînement, *créé si besoin*.
- **train.metrics.base_dir** : chemin d'accès au dossier contenant les métriques calculées, *créé si besoin*.
- **train.metrics.test_confusion_matrix** : chemin d'accès à la matrice de confusion sur l'ensemble de test, *créé si besoin*.
- **train.metrics.test_report** : chemin d'accès au fichier **.json** contenant les métriques additionnelles sur l'ensemble de test pour chaque classe et globalement, *créé si besoin*.
- **train.records.base_dir** : chemin d'accès au dossier contenant les résultats sous forme de graphiques, *créé si besoin*.
- **train.records.fit_plots** : chemin d'accès au fichier **.jpg** des courbes d'apprentissage, *créé si besoin*.
- **train.records.class_validation_plots_dir** : chemin d'accès au dossier contenant les courbes d'apprentissage pour chaque classe, *créé si besoin*.
- **train.records.validation_plots** : chemin d'accès au fichier **.jpg** des courbes d'apprentissage sur l'ensemble de validation, *créé si besoin*.

## Lancement de l'entraînement

L'exécution de l'entraînement se fait avec la commande suivante à la racine du projet :

```shell
    python -m src.models.TLModel train <FLAG>
```

L'argument ```<FLAG>``` peut être remplacé par ```--no-test``` si l'on souhaite entraîner le modèle sans le tester. Dans ce cas, certaines informations sur certaines métriques ne seront pas produites.

# Inférence du modèle de transfer learning

L'inférence du modèle se fait par l'intermédiaire du fichier de configuration disponible et éditable [ici](../../src/models/TLModel/model_config.yaml). La section concernée est celle dont la clé principale est `predict`.

Il est nécessaire d'avoir exécuté en amont l'entraînement du modèle.

L'inférence permet d'intégrer des résultats d'interprétabilité (Grad-CAM et extractions de caractéristiques) en fonction de la configuration choisie.

## Configuration

La configuration est exposée sous format **yaml**. On y trouvera les éléments suivants au sein de la clé `predict` :

- **model_path** : le chemin d'accès au fichier **.keras** contenant le modèle utilisé pour l'inférence (coïncide en général avec le champ **best_model** du bloc `train`).
- **config** : les paramètres généraux spécifiques à l'inférence.
- **input** : les différents chemins d'accès aux dossiers où chercher les images d'entrée. Peut être une image du jeu de test ou une image ajoutée dans le dossier dédié `input.new_data_dir`.

### Chemins

L'inférence nécessite l'accès à différents fichiers et répertoires (pré-existants ou non) :

- **predict.config.logs.file_path** : chemin d'acès au fichier de log, *créé si besoin*. Il est conseillé de conserver le préfixe `{DATE}_` pour le nom du fichier.
- **predict.model_path** : chemin d'accès au modèle **.keras** utilisé.
- **predict.input.test_data_dir** : chemin d'accès au dossier contenant les images de test.
- **predict.input.new_data_dir** : chemin d'accès au dossier contenant des images nouvelles ou non.
- **predict.output.results_dir** : chemin d'accès au dossier contenant les résultats des prédictions. Chaque exécution d'une prédiction génère un fichier **.json** contenant les valeurs des prédictions, les chemins d'accès aux différents résultats d'interprétabilité et des méta-données.
- **predict.output.interpretability_dir** : chemin d'accès au dossier contenant les résultats liés à l'interprétabilité. Un dossier du même nom que le fichier **.json** généré via `predict.output.results_dir` est créé dans lequel les différents résultats d'interprétabilité peuvent être trouvés.

## Lancement de l'inférence

L'exécution de l'inférence se fait avec la commande suivante à la racine du projet :

```shell
    python -m src.models.TLModel predict [IMAGES]
```

L'argument `IMAGES` permet de renseigner le nom des images si l'on souhaite réaliser une prédiction sur des images spécifiques. Dans ce cas, les images doivent être trouvées dans `predict.input.new_data_dir` ou dans `predict.input.test_data_dir` du fichier de configuration. L'argument est ***optionnel*** ; lorsque la commande est exécutée sans renseigner de noms d'images, toutes les images présentes dans `predict.input.new_data_dir` seront utilisées pour la prédiction.

## Description des résultats

### Fichier principal

Le fichier de sortie principal contenant à la fois les labels prédits et les informations complémentaires est le fichier **.json** dont le nom est aléatoirement généré de manière unique à chaque exécution d'inférence.

Plusieurs méta-données sont renseignées ainsi que les éléments les plus importants, soient :

- **inputs** : les chemins d'accès aux images utilisées pour l'inférence, partant de la racine du projet.
- **outputs.labels** : les labels prédits par le modèle.
- **outputs.probs** : les distributions de probabilité prédites par le modèle.

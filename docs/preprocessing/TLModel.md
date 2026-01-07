# Prétraitement des images pour le modèle de transfer learning

Le modèle de transfer learning est basé sur un modèle ResNet appliqué aux images du dataset. Le prétraitement des images est réalisé essentiellement en 2 parties :

- Augmentation des images par application de transformations (utilisation de `torch` et `kornia`).
- Création et structuration d'un dataset pour une compatibilité maximale avec le modèle (modèle implémenté avec Keras et TensorFlow en backend).

L'intégralité de la configuration du prétraitement est disponible et modifiable [ici](../../src/preprocessing/TLModel/preprocessing.yaml).

## Configuration

La configuration est exposée sous format **yaml**. On y trouvera les éléments suivants :

- **metadata** : nom et description de la tâche associée.
- **preprocessing** : le cœur de la configuration.
  - **config** : paramètres généraux du prétraitement.
  - **input** : fichiers et dossiers utilisés par le prétraitement.
  - **output** : fichiers et répertoires créés à l'issue du prétraitement.
  - **steps** : transformations et étapes de prétraitement.

Pour un traitement accéléré, il convient de définir **preprocessing.config.enable_cuda** = `true` et de choisir un nombre adapté au CPU pour le paramètre **preprocessing.config.threads**. Le paramètre **preprocessing.config.enable_cuda** permet d'utiliser les ressources GPU NVIDIA lorsque cette dernière est disponible.

### Chemins

Le prétraitement nécessite l'accès à différents fichiers et répertoires (pré-existants ou non) :

- **preprocessing.config.logs.file_path** : chemin d'acès au fichier de log, *créé si besoin*. Il est conseillé de conserver le préfixe `{DATE}_` pour le nom du fichier.
- **preprocessing.input.text_path** : chemin d'accès aux données textuelles brutes.
- **preprocessing.input.image_dir** : chemin d'accès au dossier contenant les images brutes.
- **preprocessing.input.image_labels** : chemin d'accès aux données des labels brutes.
- **preprocessing.output.output_dir** : chemin d'accès au dossier contenant les images traitées, *créé si besoin*.
- **preprocessing.output.train.image_dir** : chemin d'accès au dossier contenant les images d'entraînement, *créé si besoin*.
- **preprocessing.output.train.image_labels** : chemin d'accès au fichier contenant les labels d'entraînement associés aux images d'entraînement, *créé si besoin*.
- **preprocessing.output.test.image_dir** : chemin d'accès au dossier contenant les images de test, *créé si besoin*.
- **preprocessing.output.test.image_labels** : chemin d'accès au fichier contenant les labels de test associés aux images de test, *créé si besoin*.
- **preprocessing.output.dataset.dataset_dir** : chemin d'accès au dossier contenant le dataset adapté au modèle Keras/TensorFlow, *créé si besoin*.
- **preprocessing.output.dataset.train_set** : chemin d'accès au dossier contenant le dataset d'entraînement adapté au modèle Keras/TensorFlow, *créé si besoin*.
- **preprocessing.output.dataset.test_set** : chemin d'accès au dossier contenant le dataset de test adapté au modèle Keras/TensorFlow, *créé si besoin*.

### Transformation des images

Les classes suivantes peuvent être utilisées au sein du prétraitement pour réaliser l'augmentation des images. L'intégration dans le fichier de configuration se fait via le schéma suivant qui se place dans **preprocessing.steps** :

- **<step_name>**
  - **enable**, *bool* : `true` pour intégrer la transformation dans le prétraitement, `false` sinon.
  - **transformer**, *str* : nom exact de la classe qui implémente la transformation.
  - **params**:
    - **<param_name>**, *str* : **<param_value>**
    - ...

Les noms des **<param_name>** doivent respecter le nom exact des paramètres du constructeur de la classe qui implémente la transformation. Les valeurs données à **<step_name>** n'ont aucune importance autre que la compréhension des transformations.

La liste des transformations implémentées est donnée ci-dessous :

- **RandomImageRotation(degree: float, p: float)** : Applique une rotation d'angle compris entre `-degree` et `degree` à une image avec probabilité `p`.
- **RandomImageHFlip(p: float)** : Applique un retournement horizontal à une image avec probabilité `p`.
- **RandomImageVFlip(p: float)** : Applique un retournement vertical à une image avec probabilité `p`.
- **RandomImageCrop(min_scale: float, p: float)** : Applique un cropping à une image d'échelle supérieure à  `min_scale` avec probabilité `p`.
- **RandomImageZoom(min_scale: float, max_scale: float, p: float)** : Applique un zoom/dezoom à une image d'intensité comprise entre `min_scale` et `max_scale` avec probabilité `p`.
- **RandomImageBlur(max_kernel: int, p: float)** : Applique un flou gaussien de taille de noyau inférieur à `max_kernel` à une image avec probabilité `p`.
- **RandomImageNoise(max_std: float, p: float)** : Applique un bruit blanc de dispersion inférieure à `max_std` à une image avec probabilité `p`.
- **RandomImageContrast(min_factor: float, max_factor: float, p: float)** : Applique un contraste à une image d'un facteur compris entre `min_factor` et `max_factor` avec probabilité `p`.
- **RandomImageColoration(p: float)** : Applique un mapping de couleur à une image avec probabilité `p`. Le mapping est choisi aléatoirement entre un mapping en nuance de gris, un mapping négatif, une permutation aléatoire des canaux, ou un mélange linéaire aléatoire des canaux.
- **RandomImageDropout(min_area: float, max_area: float, p: float)** : Applique un dropout aléatoirement sur une image avec probabilité `p`. La zone de dropout couvre aléatoirement entre `100 * min_area` % et `100 * max_area` % de l'image.
- **RandomImagePixelDropout(max_rate: float)** : Applique un dropout de pixels uniformément sur une image avec probabilité `p`. Le taux de pixels maximum éteints est donné par `max_rate`.

## Lancement du prétraitement

L'exécution d'un prétraitement se fait avec la commande suivante à la racine du projet :

```shell
    python -m src.preprocessing.TLModel
```

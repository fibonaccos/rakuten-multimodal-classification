# Configuration du preprocessing avec config.json

## Introduction

Pour utiliser le fichier config.json dans un fichier python, il faut le charger :

```python
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))  # pour trouver le module src/

from src.config_loader import get_config  # fonction pour récupérer les infos de config.json
```
Le chemin `"../../"` peut être amené à changer selon la position du fichier python par rapport au fichier config.json. Dans l'exemple, on a :

- src/
    - preprocessing/
        - main_pipeline.py
    - config_loader.py
- config.json


On peut ensuite utiliser la configuration souhaitée (preprocessing, training, ...) :

```python
PREPROCESSING_CONFIG = get_config("PREPROCESSING")  # récupère la configuration du preprocessing sous forme de dictionnaire
```


## Preprocessing

Contient l'ensemble des paramètres et méta-données nécessaires pour la réalisation du preprocessing.

### Paths

Contient les chemins pour accéder aux données.

- **rawTextData**, ***str*** : chemin vers le dataset textuel original.
- **rawImageFolder**, ***str*** : chemin vers le dossier d'images originales.
- **rawLabels**, ***str*** : chemin vers les labels originaux.
- **cleanTextTrainData**, ***str*** : chemin vers le dataset textuel d'entraînement préparé.
- **cleanTextTestData**, ***str*** : chemin vers le dataset textuel de test préparé.
- **cleanImageTrainFolder**, ***str*** : chemin vers le dossier d'images d'entraînement préparé.
- **cleanImageTestFolder**, ***str*** : chemin vers le dossier d'images de test préparé.
- **cleanTrainLabels**, ***str*** : chemin vers les labels d'entraînement joints aux ids du dataset d'entraînement.
- **cleanTestLabels**, ***str*** : chemin vers les labels de test joints aux ids du dataset de test.

### Pipeline

Contient les paramètres et méta-données nécessaires pour l'exécution des pipelines.

- **toPipe**, ***str*** : choix du pipeline ; "text" pour exécuter le pipeline textuel, "image" pour le pipeline image et "all" pour exécuter toutes les pipelines.
- **sampleSize**, ***int*** : taille de l'échantillon à traiter. Si **sampleSize** $\small \le$ 0, toutes les données sont utilisées.
- **trainSize**, ***float*** : pourcentage de l'échantillon à utiliser comme jeu d'entraînement.
- **randomState**, ***int*** : graine aléatoire générale pour la reproductibilité.
- **textpipeline** : contient les paramètres du pipeline textuel. Les caractères indiqués dans les champs **Characters* ont vocation à être utilisés au sein d'un transformer de nettoyage de caractères.
    - **constants** :
        - **textualColumns**, ***list[str]*** : noms des colonnes du dataset sujet au traitement.
        - **alphanumCharacters**, ***str*** : caractères alphanumériques conservés lors du nettoyage.
        - **ponctuationCharacters**, ***str*** : ponctuation conservée lors du nettoyage.
        - **accentCharacters**, ***str*** : accents conservés lors du nettoyage.
        - **maxTokenVectorizer**, ***int*** : nombre maximum de tokens supportés par le modèle de vectorisation utilisé.
        - **embeddingDimension**, ***int*** : dimension des vecteurs issues de la vectorisation.
    - **steps** : étapes intégrées dans le pipeline textuel. L'ordre importe. Doit uniquement contenir des classes supportées par l'API scikit-learn : les classes doivent hériter de *BaseEstimator*, *TransformerMixin* et doivent implémenter les méthodes *fit* et *transform*. Chaque *step* contient :
        - **stepName**, ***str*** : le nom de l'étape. N'a aucun effet sur le traitement.
        - **transformer**, ***classname*** : nom de la classe utilisée pour la transformation.
        - **params**, ***dict[str, Any]*** : un dictionnaire de paramètres pour instancier la classe utilisée. Sensible à la casse.
-  **imagepipeline** : contient les paramètres du pipeline d'images.
    - **constants** :
        - **imageShape**, ***list[int]*** : les dimensions des images, en convention *channel-last*.
        - **enableCuda**, ***bool*** : permet d'utiliser le GPU pour exécuter les transformations d'images. Nécessite un GPU compatible.
        - **numThreads**, ***bool*** : nombre de threads à allouer pour le traitement CPU en parallèle.
    - **steps** : étapes intégrées dans le pipeline d'images. L'ordre importe. Doit uniquement contenir des classes héritées de *BaseImageTransform* et doivent implémenter la méthode $\,$*\_\_call\_\_* qui prend en paramètre une image sous forme de *torch.Tensor* et 2 générateurs *torch.Generator* (nécessaire au support CPU et GPU). Chaque *step* contient :
        - **stepName**, ***str*** : le nom de la transformation. Aucun effet sur le traitement.
        - **transformer**, ***classname*** : nom de la classe utilisée pour la transformation.
        - **params**, ***dict[str, Any]*** : dictionnaire de paramètres pour instancier la classe utilisée. Sensible à la casse.

## Logs

Contient les paramètres pour la configuration des logs associés aux différents composants.

- **filePath**, ***str*** : chemin vers le fichier log à créer. La paramètre *{DATE}* permet d'historiser les logs en préfixant le nom du fichier avec la date et l'heure d'exécution sous la forme *YYMMDD-HHMMSS_* + *name.log*
- **baseFormat**, ***str*** : permet de formater les informations précédant le contenu d'un message loggé.
- **dateFormat**, ***str*** : permet de définir le format de la date lors d'un message loggé.

Pour créer un logger dédié, on utilise les instructions suivantes (pour importer le contenu de `src`, voir l'introduction) :

```python
from src.logger import build_logger


LOG_CONFIG = get_config("LOGS")  # charge le contenu de la config "LOGS"
PIPELOGGER = build_logger(name="pipeline",
                          filepath=LOG_CONFIG["filePath"],
                          baseformat=LOG_CONFIG["baseFormat"],
                          dateformat=LOG_CONFIG["dateFormat"],
                          level=logging.INFO)
```

Dans l'exemple, `PIPELOGGER` écrit dans le fichier `YYMMDD-HHMMSS_pipeline.log` où les informations temporelles sont remplacées par le moment où `get_config` est appelé.

## Classes implémentées

On recense l'ensemble des classes implémentées pour le preprocessing.

### Transformations textuelles

- `CharacterCleaner()` : Adapté à la vectorisation par des modèles de transformers, utilise les champs **Characters* de config.json pour le nettoyage.
- `Vectorizer(model: str)` : Vectorisation en utilisant un `model` provenant du module `sentence-transformers`. Nécessite de paramétrer correctement le champs `maxTokenVectorizer` et `embeddingDimension` suivant le modèle utilisé.
- `CosineImputer(n_neighbors: int, excluded_cols: list[str])` : Impute les valeurs manquantes en utilisant `KNNImputer` du module `sklearn.impute`.
- `EmbeddingScaler(scaling: str, excluded_cols: list[str])` : Scale the data using module `sklearn`. Parameter `scaling` can produce a `StandardScaler`, `RobustScaler`, or `MinMaxScaler`.

### Transformations d'images

- `RandomImageRotation(degree: float, p: float)` : Applique une rotation d'angle compris entre `-degree` et `degree` à une image avec probabilité `p`.
- `RandomImageHFlip(p: float)` : Applique un retournement horizontal à une image avec probabilité `p`.
- `RandomImageVFlip(p: float)` : Applique un retournement vertical à une image avec probabilité `p`.
- `RandomImageCrop(min_scale: float, p: float)` : Applique un cropping à une image d'échelle supérieure à  `min_scale` avec probabilité `p`.
- `RandomImageZoom(min_scale: float, max_scale: float, p: float)` : Applique un zoom/dezoom à une image d'intensité comprise entre `min_scale` et `max_scale` avec probabilité `p`.
- `RandomImageBlur(max_kernel: int, p: float)` : Applique un flou gaussien de taille de noyau inférieur à `max_kernel` à une image avec probabilité `p`.
- `RandomImageNoise(max_std: float, p: float)` : Applique un bruit blanc de dispersion inférieure à `max_std` à une image avec probabilité `p`.
- `RandomImageContrast(min_factor: float, max_factor: float, p: float)` : Applique un contraste à une image d'un facteur compris entre `min_factor` et `max_factor` avec probabilité `p`.
- `RandomImageColoration(p: float)` : Applique un mapping de couleur à une image avec probabilité `p`. Le mapping est choisi aléatoirement entre un mapping en nuance de gris, un mapping négatif, une permutation aléatoire des canaux, ou un mélange linéaire aléatoire des canaux.
- `RandomImageDropout(min_area: float, max_area: float, p: float)` : Applique un dropout aléatoirement sur une image avec probabilité `p`. La zone de dropout couvre aléatoirement entre `100 * min_area` % et `100 * max_area` % de l'image.
- `RandomImagePixelDropout(max_rate: float)` : Applique un dropout de pixels uniformément sur une image avec probabilité `p`. Le taux de pixels maximum éteints est donné par `max_rate`.

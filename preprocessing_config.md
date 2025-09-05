# Configuration du preprocessing avec config.json

## Introduction

Pour utiliser le fichier config.json dans un fichier python, il faut le charger :

```python
import sys
import os
import logging

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

### Pipeline

Contient les paramètres et méta-données nécessaires pour l'exécution des pipelines.

- **sampleSize**, ***int*** : taille de l'échantillon à traiter.
- **trainSize**, ***float***: pourcentage de l'échantillon à utiliser comme jeu d'entraînement.
- **randomState**, ***int***: graine aléatoire générale pour la reproductibilité.
- **multithread**, ***bool***: permet d'exécuter en parallèle les pipelines textuel et d'images.
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
        - **params**, ***dict[str, Any]*** : un dictionnaire de paramètres pour instancier la classe utilisée.
-  **imagepipeline** : contient les paramètres du pipeline d'images.
    - **constants** :
        - **imageShape**, ***list[int]*** : les dimensions des images, en convention *channel-last*.
        - **batchSize**, ***int*** : taille des batchs pour le traitement par lots.
        - **multithread**, ***bool*** : permet d'exécuter en parallèle certaines parties du pipeline (chargement, sauvegarde des images notamment).
    - **steps** : étapes intégrées dans le pipeline d'images. L'ordre importe. Doit uniquement contenir des classes héritées de *nn.Module* de la librairie *torch*, et doivent implémenter la méthode *forward*. Chaque *step* contient :
        - **stepName**, ***str*** : le nom de la transformation. Aucun effet sur le traitement.
        - **transformer**, ***classname*** : nom de la classe utilisée pour la transformation.
        - **params**, ***dict[str, Any]*** : dictionnaire de paramètres pour instancier la classe utilisée.

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
- `RandomImageFlip(horizontal: bool, vertical: bool, p: float)` : Applique un retournement horizontal et/ou vertical à une image avec probabilité `p`.
- `RandomImageCrop(crop_window: list[int], p: float)` : Applique un cropping à une image de taille `(crop_window[0], crop_window[1])` avec probabilité `p`.
- `RandomImageZoom(factor: float, p: float)` : Applique un zoom/dezoom à une image d'intensité `1 + factor` avec probabilité `p`.
- `RandomImageBlur(p: float)` : Applique un flou gaussien à une image avec probabilité `p`.
- `RandomImageNoise(p: float)` : Applique un bruit blanc à une image avec probabilité `p`.
- `RandomImageContrast(factor: float, p: float)` : Applique un contraste à une image de facteur `1 + factor` avec probabilité `p`.
- `RandomImageColoration(p: float)` : Applique un mapping de couleur à une image avec probabilité `p`. Le mapping est choisi aléatoirement entre un mapping en nuance de gris, un mapping négatif, ou une permutation aléatoire des canaux.
- `RandomImageDropout(dropout: list[float], p: float)` : Applique un dropout aléatoirement sur une image avec probabilité `p`. La zone de dropout couvre aléatoirement entre `100 * dropout[0]` et `100 * dropout[1]` % de l'image.

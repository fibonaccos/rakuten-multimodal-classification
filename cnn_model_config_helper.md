# Configuration du modèle CNN

Les paramètres de configuration du CNN sont intégrés dans le fichier `config.json`.

## Récupération du fichier `config.json`

Pour utiliser le fichier de configuration, il suffit d'insérer les lignes suivantes pour récupérer le contenu sous forme de dictionnaire :

```python
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))  # pour trouver le module src/

from src.config_loader import get_config  # fonction pour récupérer les infos de config.json
```

Le chemin `"../../"` peut être amené à changer selon la position du fichier python par rapport au fichier config.json. Dans l'exemple, on a :

- src/
  - models/
    - cnn.py
  - config_loader.py
- config.json

On peut ensuite récupérer la configuration souhaitée avec :

```python
MODELS_CONFIG = get_config("MODELS")  # récupère la configuration des modèles sous forme de dictionnaire
```

Les chemins vers les dossiers contenant les données nettoyées sont accessibles avec :

```python
paths = MODELS_CONFIG["PATHS"]
```

## Paramètrage du CNN

Le CNN est construit avec `keras` et `tensorflow` en backend.

- **randomState**, ***int*** : Entier contrôlant la reproductibilité du modèle (batchs, initialisations, etc).
- **numThreads**, ***int*** : Nombre maximum de threads à utiliser pour la création du dataset (voir ci-après).
- **enableCuda**, ***bool*** : Permet l'utilisation du GPU pour l'entraînement du modèle (fortement recommandé).
- **imageShape**, ***list[int]*** : Dimensions initiale des images en convention *channel-last*.

### Dataset

Contient les chemins vers les datasets utilisés par le modèle. Les images sont réparties en différents dossiers permettant une bonne compatibilité avec `tensorflow`.

- **folderPath**, ***str*** : Le chemin de base vers les images.
- **train**, ***str*** : Le chemin vers le dossier d'entraînement. Le dossier contient un sous-dossier pour chaque classes à prédire.
- **test**, ***str*** : Le chemin vers le dossier de test. Le dossier contient un sous-dossier pour chaque classe prédite.

### Training

Contient les hyper-paramètres associés à l'entraînement du modèle.

- **imageShape**, ***list[int]*** : Dimensions des images à l'entrée du réseau.
- **batchSize**, ***int*** : Taille des batchs pour l'entraînement.
- **epochs**, ***int*** : Nombre d'époques d'entraînement.
- **optimizer**, ***str*** : Algorithme d'optimisation utilisé à l'entraînement. Le nom doit exister dans `keras`.
- **loss**, ***str*** : Fonction de perte utilisée pour l'optimisation. Le nom doit exister dans `keras`.
- **metrics**, ***list[str]*** : Liste contenant les métriques à suivre lors de l'entraînement. Les noms doivent exister dans `keras`.
- **validationSplit**, ***float*** : Fraction du jeu d'entraînement utilisée pour la validation (calcul des métriques).

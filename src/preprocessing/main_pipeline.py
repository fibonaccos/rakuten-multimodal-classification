import src.preprocessing.images_pipeline_components as ipipe
import src.preprocessing.textual_pipeline_components as tpipe
from sklearn.pipeline import Pipeline


def pipe() -> None:
    """
    The main preprocessing pipeline.
    """
    ...


def text_pipe() -> None:
    """
    The textual datasets pipeline.
    1. Copier les datasets -> ok \\
    -> Début Pipeline sklearn
    2. CharacterCleaner sur xtrain et xtest -> ok
    3. Vectorisation sur xtrain et xtest -> ok
    4. Restructurer les datasets xtrain et xtest (éclatement des embeddings en colonnes)
    5. Remplissage des valeurs manquantes
    6. Re-sampling des classes \\
    -> Fin Pipeline sklearn
    7. Renommage des classes -> ok
    8. Sauvegarde -> ok

    Rajout de process facilitée par la structure adaptative.
    """
    ...


def image_pipe() -> None:
    """
    The image datasets pipeline.
    1. Copier les datasets -> ok
    2. Pooling -> ok
    3. Réduction de canaux -> ok
    4. Sauvegarde -> ok

    Rajout de process facilitée par la structure adaptative.
    """
    ...

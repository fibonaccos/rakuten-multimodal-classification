import images_pipeline_components as ipipe
import textual_pipeline_components as tpipe
from sklearn.pipeline import Pipeline
from pathlib import Path
import pandas as pd


OUTPUT_DATA_PATH: Path = Path("/home/wsladmin/fibonaccos/projects/rakuten-multimodal-classification/data/clean/X_train.csv")


def pipe() -> None:
    """
    The main preprocessing pipeline.
    """
    return None


def text_pipe() -> None:
    """
    The textual datasets pipeline.
    1. Copier les datasets -> ok
    2. Splitter les datasets \\
    -> Début Pipeline sklearn
    3. CharacterCleaner sur xtrain et xtest -> ok
    4. Vectorisation sur xtrain et xtest -> ok
    5. Restructurer les datasets xtrain et xtest (éclatement des embeddings en colonnes)
    6. Remplissage des valeurs manquantes
    7. Re-sampling des classes \\
    -> Fin Pipeline sklearn
    8. Renommage des classes -> ok
    9. Sauvegarde -> ok
    """
    print("Reading csv ...")
    X_train = pd.read_csv(tpipe.DATASET_TEXT_PATH["xtrain"], index_col=0).iloc[:100]
    columns = X_train.columns

    pipe = Pipeline(steps=[("character_cleaning", tpipe.CharacterCleaner()),
                           ("embedding", tpipe.Vectorizer(model="paraphrase-multilingual-MiniLM-L12-v2")),
                           ("expanding", tpipe.EmbeddingExpander(cols_to_expand=tpipe.TEXTUAL_COLUMNS))])

    print("Pipeline started")
    X_train = pd.DataFrame(pipe.transform(X_train), columns=columns)
    print("Pipeline finished")

    print("Saving ...")
    X_train.to_csv(OUTPUT_DATA_PATH, index=False)
    return None


def image_pipe() -> None:
    """
    The image datasets pipeline.
    1. Copier les datasets -> ok
    2. Pooling -> ok
    3. Réduction de canaux -> ok
    4. Sauvegarde -> ok

    Rajout de process facilitée par la structure adaptative.
    """
    return None


text_pipe()

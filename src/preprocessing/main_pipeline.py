import images_pipeline_components as ipipe
import textual_pipeline_components as tpipe
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd


OUTPUT_DATA_PATH: Path = Path(__file__).cwd() / "data/clean/X_data.csv"


def pipe() -> None:
    """
    The main preprocessing pipeline.
    """
    return None


def text_pipe(train_size: float = 0.8, random_state: int = 42, nrows: int = 0) -> None:
    """
    The textual datasets pipeline.
    1. Copier les datasets -> ok
    2. Splitter les datasets \\
    -> Début Pipeline sklearn
    3. CharacterCleaner sur xtrain et xtest -> ok
    4. Vectorisation sur xtrain et xtest -> ok
    5. Restructurer les datasets xtrain et xtest (éclatement des embeddings en colonnes) -> ok
    6. Remplissage des valeurs manquantes
    7. Re-sampling des classes \\
    -> Fin Pipeline sklearn
    8. Renommage des classes -> ok
    9. Sauvegarde -> ok
    """
    print("Reading raw data ...")
    if nrows > 0:
        X_data = pd.read_csv(tpipe.DATASET_TEXT_PATH["xtrain"], index_col=0, nrows=nrows)
        y_data = pd.read_csv(tpipe.DATASET_TEXT_PATH["ytrain"], index_col=0, nrows=nrows)
    else:
        X_data = pd.read_csv(tpipe.DATASET_TEXT_PATH["xtrain"], index_col=0)
        y_data = pd.read_csv(tpipe.DATASET_TEXT_PATH["ytrain"], index_col=0)

    preparation_pipe = Pipeline(steps=[("character_cleaning", tpipe.CharacterCleaner()),
                                       ("embedding", tpipe.Vectorizer(model="paraphrase-multilingual-MiniLM-L12-v2")),
                                       ("expanding", tpipe.EmbeddingExpander(cols_to_expand=tpipe.TEXTUAL_COLUMNS))])#,
                                       #("filling_missing_values", tpipe.MissingEmbeddingFiller(mode="naive"))])

    print("Preparation pipeline started")

    X_clean = preparation_pipe.transform(X_data)

    print("Preparation pipeline finished")

    pd.DataFrame(X_clean).to_csv(OUTPUT_DATA_PATH, index=False)
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


text_pipe(random_state=42, nrows=100)

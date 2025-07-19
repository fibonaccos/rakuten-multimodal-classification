import images_pipeline_components as ipipe
import textual_pipeline_components as tpipe
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd


OUTPUT_DATA_PATH: dict[str, Path] = {"train": Path(__file__).cwd() / "data/clean/train.csv",
                                     "test": Path(__file__).cwd() / "data/clean/test.csv"}


def pipe() -> None:
    """
    The main preprocessing pipeline.
    """
    return None


def text_pipe(train_size: float = 0.8, random_state: int = 42, nrows: int = 0) -> None:
    """
    The textual datasets pipeline.
    1. Copier les datasets -> ok
    2. Splitter les datasets -> ok \\
    -> Début Pipeline sklearn
    3. CharacterCleaner sur xtrain et xtest -> ok
    4. Vectorisation sur xtrain et xtest -> ok
    5. Restructurer les datasets xtrain et xtest (éclatement des embeddings en colonnes) -> ok
    6. Remplissage des valeurs manquantes -> ok
    7. Re-sampling des classes \\
    -> Fin Pipeline sklearn
    8. Renommage des classes -> ok
    9. Sauvegarde -> ok
    """
    print("Reading raw data ...")
    if nrows > 0:
        X = pd.read_csv(tpipe.DATASET_TEXT_PATH["xtrain"], index_col=0, nrows=nrows)
        y = pd.read_csv(tpipe.DATASET_TEXT_PATH["ytrain"], index_col=0, nrows=nrows)
    else:
        X = pd.read_csv(tpipe.DATASET_TEXT_PATH["xtrain"], index_col=0)
        y = pd.read_csv(tpipe.DATASET_TEXT_PATH["ytrain"], index_col=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

    pipe = Pipeline(steps=[("character_cleaning", tpipe.CharacterCleaner()),
                           ("embedding", tpipe.Vectorizer(model="paraphrase-multilingual-MiniLM-L12-v2")),
                           ("expanding", tpipe.EmbeddingExpander(cols_to_expand=tpipe.TEXTUAL_COLUMNS)),
                           ("filling_missing_values", tpipe.NaiveDescriptionFiller()),
                           ('scaling', tpipe.EmbeddingScaler(scaling="standard", excluded_cols=['productid', 'imageid', 'labels']))])
    print("[Text] Pipeline started")
    print("[Text] Transforming train data ...")
    clean_X_train = pipe.fit_transform(X_train, y_train)

    print()
    print("[Text] Transforming test data ...")
    clean_X_test = pipe.transform(X_test)

    print()
    print("[Text] Pipeline finished")
    print("[Text] Saving data ...")

    clean_train = pd.DataFrame(clean_X_train)
    clean_train = pd.concat([clean_train, y_train], axis=1).rename(columns={'prdtypecode': 'labels'})
    clean_test = pd.DataFrame(clean_X_test)
    clean_test = pd.concat([clean_test, y_test], axis=1).rename(columns={'prdtypecode': 'labels'})

    clean_train.to_csv(OUTPUT_DATA_PATH["train"], index=False)
    clean_test.to_csv(OUTPUT_DATA_PATH["test"], index=False)

    print("Preprocessing done.")
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


text_pipe(nrows=500, random_state=42)

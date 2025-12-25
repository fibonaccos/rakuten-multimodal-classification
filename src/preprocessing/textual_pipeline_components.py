from __future__ import annotations

import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config
from src.logger import build_logger


TPIPE_CONFIG = get_config("PREPROCESSING")["PIPELINE"]["TEXTPIPELINE"]
LOG_CONFIG = get_config("LOGS")
TPIPELOGGER = build_logger(name="textual_pipeline_components",
                           filepath=LOG_CONFIG["filePath"],
                           baseformat=LOG_CONFIG["baseFormat"],
                           dateformat=LOG_CONFIG["dateFormat"],
                           level=logging.INFO)

TPIPELOGGER.info("Running textual_pipeline_components.py")

TPIPELOGGER.info("Importing built-ins")

from typing import Any, Literal
import re

TPIPELOGGER.info("Imorting sklearn components")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import cosine_similarity

TPIPELOGGER.info("Importing beautifulsoup, numpy, pandas")

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

TPIPELOGGER.info("Importing sentence_transformers")

from sentence_transformers import SentenceTransformer


__all__ = ["TPIPELOGGER",
           "CharacterCleaner",
           "Vectorizer",
           "CosineImputer",
           "EmbeddingScaler"]


TEXTUAL_COLUMNS: list[str] = TPIPE_CONFIG["CONSTANTS"]["textualColumns"]

ALPHANUM_CHARACTERS: list[int] = [ord(c) for c in TPIPE_CONFIG["CONSTANTS"]["alphanumCharacters"]]
PONCTUATION_CHARACTERS: list[int] = [ord(c) for c in TPIPE_CONFIG["CONSTANTS"]["ponctuationCharacters"]]
ACCENTS_CHARACTERS: list[int] =  [ord(c) for c in TPIPE_CONFIG["CONSTANTS"]["accentCharacters"]]

ACCEPTED_CHARACTERS: list[int] = ALPHANUM_CHARACTERS + PONCTUATION_CHARACTERS + ACCENTS_CHARACTERS

MAX_TOKENS_VECTORIZER: int = TPIPE_CONFIG["CONSTANTS"]["maxTokenVectorizer"]
EMBEDDING_DIM: int = TPIPE_CONFIG["CONSTANTS"]["embeddingDimension"]


class CharacterCleaner(BaseEstimator, TransformerMixin):
    """
    A transformer to clean textual data and keep characters from `ACCEPTED_CHARACTERS`.
    """

    def __init__(self, /) -> None:
        """
        Create a `CharacterCleaner` instance.

        Returns:
            CharacterCleaner: A new instance of `CharacterCleaner`.
        """

        super().__init__()
        self.is_fitted: bool = False
        self.accepted_characters_: list[int] = ACCEPTED_CHARACTERS
        return None

    def fit(self, /, X: pd.DataFrame, y: Any = None) -> CharacterCleaner:
        """
        Does nothing. Here for `sklearn` API compatibility only.

        Args:
            X (pd.DataFrame): The variables dataset.
            y (Any, optional): The predictions dataset. Defaults to None.

        Returns:
            CharacterCleaner: The fitted transformer.
        """
        return self 

    def transform(self, /, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply to the textual columns of the dataset the following transformations :
        - `_clean_html`
        - `_keep_accepted_characters`
        - `_clean_text_format`

        These are methods of the class `CharacterCleaner`. See their documentations for more details.
        The textual columns match those found in `TEXTUAL_COLUMNS`.

        Args:
            X (pd.DataFrame): The dataframe to transform.

        Returns:
            pd.DataFrame: A new dataframe with clean textual columns.
        """

        X_transformed: pd.DataFrame = X.copy(deep=True)
        for col in TEXTUAL_COLUMNS:
            TPIPELOGGER.info(f"CharacterCleaner : cleaning {col} ({"train" if not self.is_fitted else "test"} data)")
            X_transformed[col] = X_transformed[col].apply(self._clean_html)
            X_transformed[col] = X_transformed[col].apply(self._keep_accepted_characters)
            X_transformed[col] = X_transformed[col].apply(self._clean_text_format)
        self.is_fitted = True
        return X_transformed

    def _clean_html(self, /, text: Any) -> Any:
        """
        Use `BeautifulSoup4` HTML parser to clean a text from its HTML components. Applied in the
        pipeline on each row of the textual columns. If `text` comes from a `pd.DataFrame`, it ignores
        missing values.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """

        try:
            if isinstance(text, str):
                return BeautifulSoup(text.strip(), "html.parser").get_text(separator=" ").strip()
        except Exception as e:
            TPIPELOGGER.error(f"CharacterCleaner : _clean_html - {e}")
            exit(1)
        return text

    def _keep_accepted_characters(self, /, text: Any) -> Any:
        """
        Remove all characters from a text that is not found in `ACCEPTED_CHARACTERS`. Applied in the
        pipeline on each row of the textual columns. If `text` comes from a `pd.DataFrame`, it ignores
        missing values.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """

        try:
            if isinstance(text, str):
                return ''.join(c for c in text if ord(c) in self.accepted_characters_).strip()
        except Exception as e:
            TPIPELOGGER.error(f"CharacterCleaner : _keep_accepted_characters - {e}")
            exit(1)
        return text

    def _clean_text_format(self, /, text: Any) -> Any:
        """
        Remove some unconvenient patterns from a text with regex : overspacing, mail address, links, etc.
        If `text` comes from a `pd.DataFrame`, it ignores missing values.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """

        try:
            if isinstance(text, str):
                text = text.replace('\t', ' ')
                text = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', '', text)
                text = re.sub(r'https?://\S+|www\.\S+', '', text)
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
        except Exception as e:
            TPIPELOGGER.error(f"CharacterCleaner : _clean_text_format - {e}")
            exit(1)
        return text


class Vectorizer(BaseEstimator, TransformerMixin):
    """
    A transformer to vectorize textual dataset using a `SentenceTransformer` model. As the textual
    data (especially the *descripion* column) may contains more than the supported number of tokens,
    the vectorization is eventually computed on a segmentation of *n* blocks of the text. It leads
    to *n* vectors representing the text aggregated by mean.
    """

    def __init__(self, /,
                 model: Literal["paraphrase-multilingual-MiniLM-L12-v2",
                                "distiluse-base-multilingual-cased-v1",
                                "paraphrase-multilingual-mpnet-base-v2"]) -> None:
        """
        Create a `Vectorizer` instance.

        Args:
            model (Literal, optional): The model to use. The model **paraphrase-multilingual-MiniLM-L12-v2**
                is recommended (space dimension : 384). The model **distiluse-base-multilingual-cased-v1** is
                good but may be heavy (space dimension : 512). The model **paraphrase-multilingual-mpnet-base-v2**
                is very effective but slow (space dimension : 768).

        Returns:
            Vectorizer: A new instance of `Vectorizer`.
        """

        super().__init__()
        self.model_: SentenceTransformer = SentenceTransformer(model)
        self.tokenizer_: Any = self.model_.tokenizer
        self.is_fitted: bool = False
        return None

    def fit(self, /, X: pd.DataFrame, y: Any = None) -> Vectorizer:
        """
        Does nothing. Here for `sklearn` API compatibility only.

        Args:
            X (pd.DataFrame): The variables dataset.
            y (Any, optional): The predictions dataset. Defaults to None.

        Returns:
            Vectorizer: The fitted transformer.
        """
        return self

    def transform(self, /, X: pd.DataFrame) -> pd.DataFrame:
        """
        Build embeddings of the textual columns of `X` using a prefitted `SentenceTransformer` model. 

        Args:
            X (pd.DataFrame): The dataframe to transform.

        Returns:
            pd.DataFrame: A new dataframe with embedded textual columns.
        """

        try:
            X_transformed: pd.DataFrame = X.copy(deep=True)
            for col in TEXTUAL_COLUMNS:
                TPIPELOGGER.info(f"Vectorizer : encoding {col}  ({"train" if not self.is_fitted else "test"} data)")
                X_transformed = self._vectorize(X_transformed, col)
            self.is_fitted = True
            return X_transformed
        except Exception as e:
            TPIPELOGGER.error(f"Vectorizer : transform - {e}")
            exit(1)

    def _encode(self, /, text: Any) -> Any:
        # ESSAYER D'OPTIMISER CETTE FONCTION
        """
        Encode a text using `self.model_`.

        Args:
            text (Any): The text to encode. 

        Returns:
            Any: If `text` is a missing value, returns a list of `None` values of length of embeddings. If
                `text` is a string value, returns a `np.ndarray` of shape (*embedding_dim*, ).
        """

        if not isinstance(text, str) or text.strip() == "":
            return [None for _ in range(EMBEDDING_DIM)]

        tokens = self.tokenizer_.encode(text, add_special_tokens=True, truncation=False)

        if len(tokens) <= MAX_TOKENS_VECTORIZER:
            return self.model_.encode([text])[0]
        else:
            embeddings = []
            chunks = text.split('.')
            if max([len(self.tokenizer_.encode(t, add_special_tokens=True, truncation=False)) for t in chunks]) > MAX_TOKENS_VECTORIZER:
                chunks = text.split(' ')
                for i in range(len(chunks)):
                    agg_chunk = ""
                    while (i < len(chunks)) and (len(self.tokenizer_.encode(agg_chunk, add_special_tokens=True, truncation=False)) <= MAX_TOKENS_VECTORIZER):
                        agg_chunk += " " + chunks[i]
                        agg_chunk = agg_chunk.strip()
                        i += 1
                    while len(self.tokenizer_.encode(agg_chunk, add_special_tokens=True, truncation=False)) > MAX_TOKENS_VECTORIZER:
                        agg_chunk = " ".join(agg_chunk.split(" ")[:-1]).strip()
                    embeddings.append(self.model_.encode([agg_chunk])[0])
            else:
                for chunk in chunks:
                    embeddings.append(self.model_.encode([chunk])[0])
            return np.mean(embeddings, axis=0)

    def _vectorize(self, /, X: pd.DataFrame, col: str) -> pd.DataFrame:
        embeddings = X[col].apply(lambda text: self._encode(text))
        expanded_X = pd.DataFrame(embeddings.tolist(), columns=[f"{col}_{i + 1}" for i in range(EMBEDDING_DIM)], index=X.index)
        return pd.concat([X, expanded_X], axis=1)


def compute_cosine_sim(x: np.ndarray, y: np.ndarray) -> float:
    mask = (np.isnan(x)) | (np.isnan(y))
    if mask.sum() != 0:
        return np.nan
    return 1. - cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0, 0]


class CosineImputer(BaseEstimator, TransformerMixin):
    """
    A transformer for imputing missing values using KNN algorithm. It is a wrapper of the `KNNImputer` class from
    `sklearn` used with a cosine_similarity metric to better adapt the high dimensional embeddings.
    """

    def __init__(self, /, n_neighbors: int, excluded_cols: list[str]) -> None:
        """

        Args:
            n_neighbors (int): The number of neighbors to for imputing.
            excluded_cols (list[str]): The useless columns for the transformation. Should be all columns which
                are not a part of embeddings.

        Returns:
            CosineImputer: A new instance of `CosineImputer`.
        """

        super().__init__()
        self.excluded_cols_: list[str] = excluded_cols
        self.used_cols_: list[str] = []
        self.metric_ = lambda x, y, *, missing_values=np.nan: compute_cosine_sim(x, y)
        self.imputer_: KNNImputer = KNNImputer(n_neighbors=n_neighbors, metric=self.metric_)
        self.is_fitted_: bool = False
        return None

    def fit(self, /, X: pd.DataFrame, y: Any = None) -> CosineImputer:
        """
        Fit the algorithm on the right columns.

        Args:
            X (pd.DataFrame): The dataframe containing the missing values.
            y (Any, optional): Unused. Here for API consistency only.

        Returns:
            CosineImputer: The fitted transformer.
        """

        self.used_cols_ = [col for col in X.columns if col not in self.excluded_cols_]
        self.imputer_.fit(X[self.used_cols_], y)
        return self

    def transform(self, /, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe `X` using the fitted `CosineImputer`.

        Args:
            X (pd.DataFrame): The dataframe to transform.

        Returns:
            pd.DataFrame: The transformed dataframe without missing values.
        """

        try:
            TPIPELOGGER.info(f"Imputing missing values ({"train" if not self.is_fitted_ else "test"} data)")
            X_transformed: pd.DataFrame = X.copy(deep=True)
            X_transformed[self.used_cols_] = self.imputer_.transform(X_transformed[self.used_cols_])
            self.is_fitted_ = True
            return X_transformed
        except Exception as e:
            TPIPELOGGER.error(f"EmbeddingScaler : transform - {e}")
            exit(1)


class EmbeddingScaler(BaseEstimator, TransformerMixin):
    """
    A transformer to scale data. It can uses `StandardScaler`, `RobustScaler` or `MinMaxScaler` from the sklearn
    preprocessing module.
    """

    def __init__(self, /, scaling: Literal["standard", "robust", "minmax"] = "standard", excluded_cols: list[str] = []) -> None:
        """
        Create an `EmbeddingScaler` instance.

        Args:
            scaling (str): The type of scaling. "standard" for `StandardScaler`, "robust" for `RobustScaler`
                or "minmax" for `MinMaxScaler`.
            excluded_cols (list[str]): The columns to keep unchanged.

        Returns:
            EmbeddingScaler: A new instance of `EmbeddingScaler`.
        """

        super().__init__()
        self.scaling_type_: Literal["standard", "robust", "minmax"] = scaling
        self.excluded_cols_: list[str] = excluded_cols
        self.used_cols_: list[str] = []
        self.is_fitted: bool = False
        if scaling == "robust":
            from sklearn.preprocessing import RobustScaler
            self.scaler_ = RobustScaler()
        elif scaling == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler_ = MinMaxScaler()
        else:
            from sklearn.preprocessing import StandardScaler
            self.scaler_ = StandardScaler()
        return None

    def fit(self, /, X: pd.DataFrame, y: Any = None) -> EmbeddingScaler:
        """
        Fit the scaler to the data. Must be used on the train set.

        Args:
            X (pd.DataFrame): The training variables dataset.
            y (Any, optional): The training predictions dataset. Defaults to None.

        Returns:
            EmbeddingScaler: The fitted transformer.
        """

        self.used_cols_ = [col for col in X.columns if col not in self.excluded_cols_]
        self.scaler_.fit(X[self.used_cols_], y)
        return self

    def transform(self, /, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the data using the given sklearn scaler. Does not apply on the `excluded_cols` columns.

        Args:
            X (pd.DataFrame): The dataframe to scale.

        Returns:
            pd.DataFrame: The scaled dataframe.
        """

        TPIPELOGGER.info(f"Scaling embeddings ({"train" if not self.is_fitted else "test"} data)")
        try:
            X_transformed: pd.DataFrame = X.copy(deep=True)
            X_transformed[self.used_cols_] = self.scaler_.transform(X_transformed[self.used_cols_])
            self.is_fitted = True
            return X_transformed
        except Exception as e:
            TPIPELOGGER.error(f"EmbeddingScaler : transform - {e}")
            exit(1)

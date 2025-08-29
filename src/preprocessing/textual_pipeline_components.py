from __future__ import annotations
from typing import Any, Literal
from sklearn.base import BaseEstimator, TransformerMixin
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import re
import nltk

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config_loader import get_config


__all__ = ["CharacterCleaner",
           "Vectorizer",
           "EmbeddingExpander"]


PREPROCESSING_CONFIG = get_config("PREPROCESSING")

TEXTUAL_COLUMNS: list[str] = PREPROCESSING_CONFIG["CONSTANTS"]["textualColumns"]

ALPHANUM_CHARACTERS: list[int] = [ord(c) for c in PREPROCESSING_CONFIG["CONSTANTS"]["alphanumCharacters"]]
PONCTUATION_CHARACTERS: list[int] = [ord(c) for c in PREPROCESSING_CONFIG["CONSTANTS"]["ponctuationCharacters"]]
ACCENTS_CHARACTERS: list[int] =  [ord(c) for c in PREPROCESSING_CONFIG["CONSTANTS"]["accentCharacters"]]

ACCEPTED_CHARACTERS: list[int] = ALPHANUM_CHARACTERS + PONCTUATION_CHARACTERS + ACCENTS_CHARACTERS

MAX_TOKENS_VECTORIZER: int = PREPROCESSING_CONFIG["CONSTANTS"]["maxTokenVectorizer"]


class CharacterCleaner(BaseEstimator, TransformerMixin):
    """
    A transformer to clean textual data and keep characters from `ACCEPTED_CHARACTERS`.
    """
    def __init__(self) -> None:
        """
        Create a `CharacterCleaner` instance.

        Returns:
            CharacterCleaner: A new instance of `CharacterCleaner`.
        """
        super().__init__()
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
        for i, col in enumerate(TEXTUAL_COLUMNS):
            print(f"\r[CharacterCleaner] - {i + 1}/{len(TEXTUAL_COLUMNS)}", end='')
            X_transformed[col] = X_transformed[col].apply(self._clean_html)
            X_transformed[col] = X_transformed[col].apply(self._keep_accepted_characters)
            X_transformed[col] = X_transformed[col].apply(self._clean_text_format)

        print("\r[CharacterCleaner] - DONE    ")
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
        if isinstance(text, str):
            return BeautifulSoup(text.strip(), "html.parser").get_text(separator=" ").strip()
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
        if isinstance(text, str):
            return ''.join(c for c in text if ord(c) in self.accepted_characters_)
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
        if isinstance(text, str):
            text = text.replace('\t', ' ')
            text = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', '', text)
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        return text


class Vectorizer(BaseEstimator, TransformerMixin):
    """
    A transformer to vectorize textual dataset using a `SentenceTransformer` model. As the textual
    data (especially the *descripion* column) may contains more than 512 tokens, the vectorization
    is eventually computed on a segmentation of *n* blocks of the text. It leads to *n* vectors
    representing the text that have to be aggregated. They can be aggregated by mean or by their
    maximum value.
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
        nltk.download("punkt")
        nltk.download('punkt_tab')
        self.max_chars_: int = MAX_TOKENS_VECTORIZER
        self.model_: SentenceTransformer = SentenceTransformer(model)
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
        X_transformed: pd.DataFrame = X.copy(deep=True)
        for i, col in enumerate(TEXTUAL_COLUMNS):
            print(f"\r[Vectorizer] - {i + 1}/{len(TEXTUAL_COLUMNS)}", end='')
            X_transformed[col] = X_transformed[col].apply(self._encoder)
        print("\r[Vectorizer] - DONE    ")
        return X_transformed

    def _split_text(self, /, text: str) -> list[str]:
        """
        Splits a text to a list of subtexts containing less than `MAX_TOKENS_VECTORIZER` tokens.

        Args:
            text (str): The text to split.

        Returns:
            list[str]: The splitted text with less than `MAX_TOKENS_VECTORIZER` tokens for each
                subtext.
        """
        sentences: list[str] = sent_tokenize(text, language="french")
        chunks: list[str] = []
        current: str = ""
        for sent in sentences:
            if len(current) + len(sent) < self.max_chars_:
                current += " " + sent
            else:
                chunks.append(current.strip())
                current = sent
        if current:
            chunks.append(current.strip())
        return chunks

    def _encoder(self, /, text: Any) -> Any:
        """
        Create an embedding of a text. Ignores missing values.

        Args:
            text (str | Any): The text to vectorize.

        Returns:
            np.ndarray | Any: The result of embedding if `text` is not a missing value, else returns
                the `text` unchanged.
        """
        if isinstance(text, str):
            return np.mean(self.model_.encode(self._split_text(text), convert_to_numpy=True), axis=0)
        return text


class EmbeddingExpander(BaseEstimator, TransformerMixin):
    """
    A transformer to expand embedded columns passed into the `Vectorizer` transformer. If multiple columns
    have been embedded, it expands them as `column_name_i` where `i` is the i-th dimension of the embedding.
    """
    def __init__(self, /) -> None:
        """
        Create an `EmbeddingExpander` instance.

        Returns:
            EmbeddingExpander: A new instance of `EmbeddingExpander`.
        """
        super().__init__()
        self.cols_to_expand_: list[str] = TEXTUAL_COLUMNS
        self.output_shape_: tuple[int, int] = (0, 0)
        return None

    def fit(self, /, X: pd.DataFrame, y: Any = None) -> EmbeddingExpander:
        """
        Does nothing. Here for `sklearn` API compatibility only.

        Args:
            X (pd.DataFrame): The variables dataset.
            y (Any, optional): The predictions dataset. Defaults to None.

        Returns:
            EmbeddingExpander: The fitted transformer.
        """
        return self

    def transform(self, /, X: pd.DataFrame) -> pd.DataFrame:
        """
        Expand the embedded columns provided by `cols_to_expand` attribute. Each column in `cols_to_expand`
        must contains vectors in either string or numpy arrays format. Missing values are filled with
        zero-like vectors.

        Args:
            X (pd.DataFrame): The dataframe to transform.

        Returns:
            pd.DataFrame: A new dataframe with expanded columns.
        """
        X_transformed: pd.DataFrame = X.copy(deep=True)
        expanded_cols_names: dict[str, list[str]] = {col: [] for col in TEXTUAL_COLUMNS}
        for i, col in enumerate(self.cols_to_expand_):
            print(f"\r[EmbeddingExpander] - {i + 1}/{len(self.cols_to_expand_)}", end='')

            expand_size = len(X_transformed[~X_transformed[col].isna()].reset_index()[col][0])
            X_transformed[col] = X_transformed[col].fillna(str(np.zeros(expand_size)))
            X_transformed[col] = X_transformed[col].apply(self._make_array_like)

            expanded_cols_names[col] = [col + "_" + str(i + 1) for i in range(expand_size)]

            vectors = np.stack([v for v in X_transformed[col]])
            expand_df = pd.DataFrame(vectors, columns=expanded_cols_names[col], index=X.index)
            X_transformed = pd.concat([X_transformed, expand_df], axis=1)
            X_transformed[col + "_" + "norm"] = X_transformed[expanded_cols_names[col]].apply(lambda row: np.abs(row.astype('float').values).sum(), axis=1)

        X_transformed = X_transformed.drop(columns=self.cols_to_expand_)
        self.output_shape_ = X_transformed.shape

        print(f"\r[EmbeddingExpander] - DONE    ")
        return X_transformed

    def _make_array_like(self, /, encoded_text: Any) -> Any:
        """
        Preparation task before expanding the columns. Applied through the `transform` operation on each row of
        each `cols_to_expand`'s column. Ignores missing values.

        Args:
            encoded_text (Any): Value to clean before expanding.

        Returns:
            Any: Cleaned value ready for expanding or `NaN`.
        """
        if isinstance(encoded_text, str):
            s = encoded_text.replace('\n', '').split('[')[-1].split(']')[0].split(' ')
            s = [v for v in s if v != '']
            return s
        return encoded_text


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
        self.scaler_.fit(X[self.used_cols_])
        return self

    def transform(self, /, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the data using the given sklearn scaler. Does not apply on the `excluded_cols` columns.

        Args:
            X (pd.DataFrame): The dataframe to scale.

        Returns:
            pd.DataFrame: The scaled dataframe.
        """
        X_transformed: pd.DataFrame = X.copy(deep=True)
        print(f"\r[EmbeddingScaler] ...", end='')
        X_transformed[self.used_cols_] = self.scaler_.transform(X_transformed[self.used_cols_])
        print(f"\r[EmbeddingScaler] - DONE")
        return X_transformed

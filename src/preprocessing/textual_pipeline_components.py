from __future__ import annotations
from typing import Any, Literal
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd


__all__ = ["DATASET_TEXT_PATH",
           "TEXTUAL_COLUMNS",
           "ACCEPTED_CHARACTERS",
           "MAX_TOKENS_VECTORIZER",
           "CharacterCleaner",
           "Vectorizer"]


DATASET_TEXT_PATH: dict[str, Path] = {"xtrain": Path(), "xtest": Path(), "ytrain": Path(), "ytest": Path()}
""" Paths to local textual datasets (keyed as "xtrain", "xtest", "ytrain", "ytest") given as a dict. """


TEXTUAL_COLUMNS: list[str] = ["designation", "description"]
""" Name of textual columns to preprocess found in the original datasets. """


# Lists of characters given as Unicode int that we may want to keep
ALPHANUM_CHARACTERS: list[int] = [ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"]
PONCTUATION_CHARACTERS: list[int] = [ord(c) for c in "!\"$%&'()+,-./\\:;?[]^_`|~"]
ACCENTS_CHARACTERS: list[int] =  [ord(c) for c in 'àâäçéèêëîïôöùûüÿœæ']

ACCEPTED_CHARACTERS: list[int] = ALPHANUM_CHARACTERS + PONCTUATION_CHARACTERS + ACCENTS_CHARACTERS
""" List of all the characters to keep encoded in integers. """


MAX_TOKENS_VECTORIZER: int = 500
""" The maximum number of tokens to use in `Vectorizer`. Depends on the model used. """


class CharacterCleaner(BaseEstimator, TransformerMixin):
    """
    A transformer to clean textual data and keep characters from `ACCEPTED_CHARACTERS`. It inherits from `sklearn`
    objects to have compatibility with its `Pipeline` object.
    """
    def __init__(self) -> None:
        super().__init__()
        self.accepted_characters_: list[int] = ACCEPTED_CHARACTERS
        return None

    def fit(self, X: pd.DataFrame, y: Any = None) -> CharacterCleaner:
        """
        Does nothing. Here for `sklearn` API compatibility only.

        Args:
            X (pd.DataFrame): The variables dataset.
            y (Any, optional): The predictions dataset. Defaults to None.

        Returns:
            CharacterCleaner: The fitted transformer.
        """
        return self 

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply to the textual columns of the dataset the following transformations :
        - `_clean_html`
        - `_clean_xml`
        - `_keep_accepted_characters`

        These are methods of the class `CharacterCleaner`. See their documentations for more details.
        The textual columns match those found in `TEXTUAL_COLUMNS`.

        Args:
            X (pd.DataFrame): The dataset to transform.

        Returns:
            pd.DataFrame: The dataset with clean textual columns.
        """
        X_transformed: pd.DataFrame = X.copy(deep=True)
        for col in TEXTUAL_COLUMNS:
            X_transformed[col].apply(lambda text: self._clean_html(text), inplace=True)
            X_transformed[col].apply(lambda text: self._clean_xml(text), inplace=True)
            X_transformed[col].apply(lambda text: self._keep_accepted_characters(text), inplace=True)
            X_transformed[col].apply(lambda text: self._clean_text_format(text), inplace=True)
        return X_transformed

    def _clean_html(self, text: Any) -> Any:
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

    def _clean_xml(self, text: Any) -> Any:
        """
        Use `BeautifulSoup4` XML parser to clean a text from its XML components. Applied in the
        pipeline on each row of the textual columns. If `text` comes from a `pd.DataFrame`, it
        ignores missing values.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        if isinstance(text, str):
            return BeautifulSoup(text.strip(), "lxml").get_text(separator=" ").strip()
        return text

    def _keep_accepted_characters(self, text: Any) -> Any:
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

    def _clean_text_format(self, text: Any) -> Any:
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
    def __init__(self,
                 model: Literal["paraphrase-multilingual-MiniLM-L12-v2",
                                "distiluse-base-multilingual-cased-v1",
                                "paraphrase-multilingual-mpnet-base-v2"],
                 aggregate: Literal["mean", "max"] = "mean") -> None:
        """
        Create a `Vectorizer` instance.

        Args:
            model (Literal, optional): The model to use. The model **paraphrase-multilingual-MiniLM-L12-v2**
                is recommended (space dimension : 384). The model **distiluse-base-multilingual-cased-v1** is
                good but may be heavy (space dimension : 512). The model **paraphrase-multilingual-mpnet-base-v2**
                is very effective but slow (space dimension : 768).
            combine_method (Literal[&quot;mean&quot;, &quot;max&quot;], optional): The combining
                method for vectorization. Defaults to "mean".
        """
        super().__init__()
        nltk.download("punkt")
        self.max_chars_: int = MAX_TOKENS_VECTORIZER
        self.aggregate_method_: Literal["mean", "max"] = aggregate
        self.model_: SentenceTransformer = SentenceTransformer(model)
        return None

    def fit(self, X: pd.DataFrame, y: Any = None) -> Vectorizer:
        """
        Does nothing. Here for `sklearn` API compatibility only.

        Args:
            X (pd.DataFrame): The variables dataset.
            y (Any, optional): The predictions dataset. Defaults to None.

        Returns:
            Vectorizer: The fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Build embeddings of the textual columns of `X`. Ignore 

        Args:
            X (pd.DataFrame): The dataset to transform.

        Returns:
            pd.DataFrame: The dataset with embedded textual columns.
        """
        X_transformed: pd.DataFrame = X.copy(deep=True)
        for col in TEXTUAL_COLUMNS:
            X_transformed[col].apply(lambda text: self._encoder(text), inplace=True)
        return X_transformed

    def _split_text(self, text: str) -> list[str]:
        """
        Splits a text to a list of subtexts containing less than `MAX_TOKENS_VECTORIZER` tokens.

        Args:
            text (str): The text to split.

        Returns:
            list[str]: The splitted text with less than `MAX_TOKENS_VECTORIZER` tokens for each
                subtext.
        """
        sentences: list[str] = sent_tokenize(text)
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

    def _encoder(self, text: Any) -> Any:
        """
        Create an embedding of a text. If `text` comes from a `pd.DataFrame`, it
        ignores missing values.

        Args:
            text (str | Any): The text to vectorize.

        Returns:
            np.ndarray | Any: The result of embedding if `text` is not a missing value
        """
        if isinstance(text, str):
            return np.mean(self.model_.encode(self._split_text(text)), axis=0)
        return text

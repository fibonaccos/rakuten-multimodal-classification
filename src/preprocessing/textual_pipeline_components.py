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


DATASET_TEXT_PATH: dict[str, Path] = {"xtrain": Path("/home/wsladmin/fibonaccos/projects/data/X_train.csv"),
                                      "xtest": Path("../../../data/X_test.csv"),
                                      "ytrain": Path("../../../data/Y_train.csv"),
                                      "ytest": Path("../../../data/Y_train.csv")}
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
    A transformer to clean textual data and keep characters from `ACCEPTED_CHARACTERS`.
    """
    def __init__(self) -> None:
        """
        Create a `CharacterCleaner` instance.

        Returns:
            `CharacterCleaner`: A new instance of `CharacterCleaner`.
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
        - `_clean_xml`
        - `_keep_accepted_characters`

        These are methods of the class `CharacterCleaner`. See their documentations for more details.
        The textual columns match those found in `TEXTUAL_COLUMNS`.

        Args:
            X (pd.DataFrame): The dataframe to transform.

        Returns:
            pd.DataFrame: A new dataframe with clean textual columns.
        """
        X_transformed: pd.DataFrame = X.copy(deep=True)
        for i, col in enumerate(TEXTUAL_COLUMNS):
            print(f"\r[CharacterCleaner] - {i + 1}/{len(TEXTUAL_COLUMNS)} : html cleaning", end='')
            X_transformed[col] = X_transformed[col].apply(self._clean_html)

            print(f"\r[CharacterCleaner] - {i + 1}/{len(TEXTUAL_COLUMNS)} : keep characters", end='')
            X_transformed[col] = X_transformed[col].apply(self._keep_accepted_characters)

            print(f"\r[CharacterCleaner] - {i + 1}/{len(TEXTUAL_COLUMNS)} : format cleaning", end='')
            X_transformed[col] = X_transformed[col].apply(self._clean_text_format)

        print("\r[CharacterCleaner] - Done\t\t\t")
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
                                "paraphrase-multilingual-mpnet-base-v2"],
                 combine_method: Literal["mean", "max"] = "mean") -> None:
        """
        Create a `Vectorizer` instance.

        Args:
            model (Literal, optional): The model to use. The model **paraphrase-multilingual-MiniLM-L12-v2**
                is recommended (space dimension : 384). The model **distiluse-base-multilingual-cased-v1** is
                good but may be heavy (space dimension : 512). The model **paraphrase-multilingual-mpnet-base-v2**
                is very effective but slow (space dimension : 768).
            combine_method (Literal[&quot;mean&quot;, &quot;max&quot;], optional): The combining
                method for vectorization. Defaults to "mean".

        Returns:
            `Vectorizer`: A new instance of `Vectorizer`.
        """
        super().__init__()
        nltk.download("punkt")
        nltk.download('punkt_tab')
        self.max_chars_: int = MAX_TOKENS_VECTORIZER
        self.combine_method_: Literal["mean", "max"] = combine_method
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
        Build embeddings of the textual columns of `X`. 

        Args:
            X (pd.DataFrame): The dataframe to transform.

        Returns:
            pd.DataFrame: A new dataframe with embedded textual columns.
        """
        X_transformed: pd.DataFrame = X.copy(deep=True)
        for i, col in enumerate(TEXTUAL_COLUMNS):
            print(f"\r[Vectorizer] - {i + 1}/{len(TEXTUAL_COLUMNS)} : embedding", end='')
            X_transformed[col] = X_transformed[col].apply(self._encoder)
        print("\r[Vectorizer] - Done\t\t\t")
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
        Create an embedding of a text. If `text` comes from a `pd.DataFrame`, it
        ignores missing values.

        Args:
            text (str | Any): The text to vectorize.

        Returns:
            np.ndarray | Any: The result of embedding if `text` is not a missing value
        """
        if isinstance(text, str):
            return np.mean(self.model_.encode(self._split_text(text), convert_to_numpy=True), axis=0)
        return text


class EmbeddingExpander(BaseEstimator, TransformerMixin):
    """
    A transformer to expand embedded columns passed into the `Vectorizer` transformer. If multiple columns
    have been embedded, it expands them as `column_name_i` where `i` is the i-th dimension of the embedding.
    """
    def __init__(self, /, cols_to_expand: list[str] = []) -> None:
        """
        Create an `EmbeddingExpander` instance.

        Args:
            cols_to_expand (list[str], optional): The columns to expand. If the provided list is empty, the
                dataframe will remain unchanged after the transformation. Defaults to [].

        Returns:
            `EmbeddingExpander`: A new instance of `EmbeddingExpander`.
        """
        super().__init__()
        self.cols_to_expand_: list[str] = cols_to_expand
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
        Expand the embedded columns provided by `cols_to_expand` attribute.

        Args:
            X (pd.DataFrame): The dataframe to transform.

        Returns:
            pd.DataFrame: A new dataframe with expanded columns.
        """
        X_transformed: pd.DataFrame = X.copy(deep=True)
        for i, col in enumerate(self.cols_to_expand_):
            print(f"\r[EmbeddingExpander] - {i + 1}/{len(self.cols_to_expand_)} : making array-like", end='')
            X_transformed[col] = X_transformed[col].apply(self._make_array_like)

            expand_size = len(X[~X[col].isna()].reset_index()[col][0])
            col_names = [col + "_" + str(i + 1) for i in range(expand_size)]

            print(f"\r[EmbeddingExpander] - {i + 1}/{len(self.cols_to_expand_)} : computing vectors", end='')

            vectors = np.stack([v if isinstance(v, list) else np.zeros(expand_size) for v in X_transformed[col]])
            print(vectors.shape)
            expand_df = pd.DataFrame(vectors, columns=col_names, index=X.index)
            print(expand_df.shape)

            print(f"\r[EmbeddingExpander] - {i + 1}/{len(self.cols_to_expand_)} : expanding dataframe", end='')

            X_transformed = pd.concat([X_transformed, expand_df], axis=1)

        X_transformed.drop(columns=self.cols_to_expand_, inplace=True)
        self.output_shape_ = X_transformed.shape

        print(f"\r[EmbeddingExpander] - Done\t\t\t")
        return X_transformed

    def _make_array_like(self, /, encoded_text: Any) -> Any:
        if isinstance(encoded_text, str):
            s = encoded_text.replace('\n', '').split('[')[-1].split(']')[0].split(' ')
            s = [v for v in s if v != '']
            return s
        return encoded_text


class MissingEmbeddingFiller(BaseEstimator, TransformerMixin):
    """
    A transformer to fill missing values from the *description* column. It fills the missing values by
    sampling from the distribution of existing values for each class.
    """
    def __init__(self, /, mode: Literal["naive", "sampling"] = "naive") -> None:
        """
        Create a `MissingEmbeddingFiller` instance.

        Args:
            mode (Literal[&quot;naive&quot;, &quot;sampling&quot;], optional): The method of filling values.
                If "naive", it will fill the missing descriptions with the corresponding titles. If "sampling",
                it will draw a sample from the available descriptions of the corresponding class. Defaults
                to "naive".

        Returns:
            MissingEmbeddingFiller: A new `MissingEmbeddingFiller` instance.
        """
        super().__init__()
        self.mode_: str = mode
        return None

    def fit(self, /, X: pd.DataFrame, y: Any = None) -> MissingEmbeddingFiller:
        """
        Does nothing. Here for `sklearn` API compatibility only.

        Args:
            X (pd.DataFrame): The variables dataset.
            y (Any, optional): The predictions dataset. Defaults to None.

        Returns:
            MissingEmbeddingFiller: The fitted transformer.
        """
        return self

    def transform(self, /, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed: pd.DataFrame = X.copy(deep=True)
        return X_transformed

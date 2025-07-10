from __future__ import annotations
from typing import Any
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from bs4 import BeautifulSoup


__all__ = []


DATASET_TEXT_PATH: dict[str, Path] = {"xtrain": Path(), "ytrain": Path(), "xtest": Path(), "ytest": Path()}

TEXTUAL_COLUMNS: list[str] = ["designation", "description"]

ALPHANUM_CHARACTERS: list[int] = [ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"]
PONCTUATION_CHARACTERS: list[int] = [ord(c) for c in "!\"$%&'()+,-./\\:;?[]^_`|~"]
ACCENTS_CHARACTERS: list[int] =  [ord(c) for c in 'àâäçéèêëîïôöùûüÿœæ']

ACCEPTED_CHARACTERS: list[int] = ALPHANUM_CHARACTERS + PONCTUATION_CHARACTERS + ACCENTS_CHARACTERS


class CharacterScanner(BaseEstimator, TransformerMixin):
    """
    A transformer to remove from data a given list of characters. It inherits from `sklearn`
    objects to have compatibility with its Pipeline object.
    """
    def __init__(self) -> None:
        self.accepted_characters_: list[int] = ACCEPTED_CHARACTERS
        return None

    def fit(self, X: Any, y: Any = None) -> CharacterScanner:
        return self 

    def transform(self, X: Any) -> Any:
        X_transformed: Any = X.copy(deep=True)
        for col in TEXTUAL_COLUMNS:
            X_transformed[col].apply(lambda s: self._clean_html(s), inplace=True)
            X_transformed[col].apply(lambda s: self._clean_xml(s), inplace=True)
            X_transformed[col].apply(lambda s: self._clean_abnormal_chars(s), inplace=True)
        return X_transformed

    def _clean_html(self, s: str) -> str:
        """
        Use `BeautifulSoup4` HTML parser to clean a string from its HTML components. Applied in the
        textual pipeline on each row of the textual columns.

        Args:
            s (str): The string to clean.

        Returns:
            str: The cleaned string.
        """
        return BeautifulSoup(s.strip(), "html.parser").get_text(separator=" ").strip()

    def _clean_xml(self, s: str) -> str:
        """
        Use `BeautifulSoup4` XML parser to clean a string from its XML components. Applied in the
        textual pipeline on each row of the textual columns.

        Args:
            s (str): The string to clean.

        Returns:
            str: The cleaned string.
        """
        return BeautifulSoup(s.strip(), "lxml").get_text(separator=" ").strip()

    def _clean_abnormal_chars(self, s: str) -> str:
        """
        Remove all characters from a string that is not found in `ACCEPTED_CHARACTERS`. Applied in the
        textual pipeline on each row of the textual columns.

        Args:
            s (str): The string to clean.

        Returns:
            str: The cleaned string.
        """
        return ''.join(c for c in s if ord(c) in self.accepted_characters_)

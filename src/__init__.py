"""
Main module of the project. Provides the following submodules :
- models : contains models implementations, training and inference pipelines.
- preprocessing : contains preprocessing components and pipelines for models.
"""

from . import models
from . import preprocessing


__all__ = [
    "models",
    "preprocessing"
]

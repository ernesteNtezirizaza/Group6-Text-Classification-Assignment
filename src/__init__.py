"""Shared utilities package for text classification project."""

__version__ = '1.0.0'
__author__ = 'Group 6'

from . import preprocessing
from . import embeddings
from . import evaluation
from . import utils

__all__ = ['preprocessing', 'embeddings', 'evaluation', 'utils']

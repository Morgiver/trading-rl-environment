"""Enumerations for trading RL environment."""

from enum import Enum, auto


class DataMode(Enum):
    """Mode for reading dataset during training."""
    LINEAR = auto()   # Sequential traversal of dataset (continuous across episodes)
    SHUFFLE = auto()  # Random day selection for each episode

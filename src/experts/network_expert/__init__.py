"""CICIDS network expert modules (parser/train/test + model helpers)."""

from src.experts.network_expert.constants import CANONICAL_CICIDS_15_CLASSES
from src.experts.network_expert.model import CNNLSTMClassifier, NetworkExpert

__all__ = ["CANONICAL_CICIDS_15_CLASSES", "CNNLSTMClassifier", "NetworkExpert"]

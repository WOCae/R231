"""汎用構造解析 GNN ツールキット"""

from .config import GNNConfig
from .data import FEADataProcessor
from .model import StructuralGNN
from .toolkit import GNNToolkit
from .ui import GNNToolkitUI

__all__ = [
    "GNNConfig",
    "FEADataProcessor",
    "StructuralGNN",
    "GNNToolkit",
    "GNNToolkitUI",
]

__version__ = "2.0.0"

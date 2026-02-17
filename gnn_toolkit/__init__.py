"""汎用構造解析 GNN ツールキット — メッシュ形状変更対応版"""

from .config import GNNConfig
from .data import FEADataProcessor
from .inp_reader import InpFileReader
from .model import StructuralGNN
from .toolkit import GNNToolkit
from .ui import GNNToolkitUI

__all__ = [
    "GNNConfig",
    "FEADataProcessor",
    "InpFileReader",
    "StructuralGNN",
    "GNNToolkit",
    "GNNToolkitUI",
]

__version__ = "3.0.0"

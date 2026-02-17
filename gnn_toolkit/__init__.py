"""汎用構造解析 GNN ツールキット — 荷重方向ベクトル対応版"""

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

__version__ = "3.1.0"

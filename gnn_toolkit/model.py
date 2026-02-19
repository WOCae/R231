"""
StructuralGNN — 汎用構造解析グラフニューラルネットワーク

Encoder → Processor (残差 SAGEConv × N 層) → Decoder (変位 / 応力ヘッド)
v3.1: 荷重方向ベクトル (3成分) 対応。入力次元を config.n_features で可変化。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class _ResBlock(nn.Module):
    """残差接続付き GraphSAGE ブロック。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = SAGEConv(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return x + F.relu(self.norm(self.conv(x, edge_index)))


class StructuralGNN(nn.Module):
    """
    汎用構造解析 GNN。

    Parameters
    ----------
    config : GNNConfig
        モデル構造・物理制約の設定を保持するオブジェクト
    """

    def __init__(self, config) -> None:
        super().__init__()
        h = config.hidden_dim
        self._load_cols = config.load_cols

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.n_features, h),
            nn.ReLU(),
            nn.Linear(h, h),
        )

        # Processor
        self.blocks = nn.ModuleList(
            [_ResBlock(h) for _ in range(config.n_layers)]
        )

        # Decoder — 変位・応力を独立ヘッドで予測
        self.disp_head = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, h // 2), nn.ReLU(),
            nn.Linear(h // 2, 3),
        )
        self.stress_head = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, h // 2), nn.ReLU(),
            nn.Linear(h // 2, 1),
        )

        self.linear_scaling: bool = config.linear_scaling

    def forward(self, data: Data) -> torch.Tensor:
        x, ei = data.x, data.edge_index

        # 荷重ベクトル (Fx, Fy, Fz) の取得
        start = self._load_cols[0]
        load_vec = data.x[:, start:start + 3]  # (N, 3)

        # スケーリング用のスカラー比率 = 荷重ベクトルのノルム最大値
        load_mag = torch.norm(load_vec, dim=1)  # (N,)
        ratio = torch.max(load_mag)

        h = self.encoder(x)
        for blk in self.blocks:
            h = blk(h, ei)

        d = self.disp_head(h)
        s = self.stress_head(h)

        if self.linear_scaling:
            d = d * ratio
            s = s * ratio

        return torch.cat([d, s], dim=1)

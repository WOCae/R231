"""
GNNConfig — 設定クラス

データから正規化定数を自動計算し、JSON で永続化する。
コンストラクタ引数で任意のパラメータをオーバーライド可能。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np


@dataclass
class GNNConfig:
    """GNN の全設定を一元管理するデータクラス"""

    # ── 単位変換 ──────────────────────────────────────────
    to_mm: float = 1000.0     # m → mm
    to_mpa: float = 1e-6      # Pa → MPa

    # ── 正規化定数（None = auto_calibrate で自動決定）──────
    norm_coord: Optional[float] = None
    norm_disp: Optional[float] = None
    norm_stress: Optional[float] = None
    train_load: float = 1000.0

    # ── モデル構造 ────────────────────────────────────────
    hidden_dim: int = 128
    n_layers: int = 4
    n_features: int = 5       # x, y, z, is_fixed, load_feat
    n_outputs: int = 4        # ux, uy, uz, stress

    # ── 学習パラメータ ────────────────────────────────────
    epochs: int = 5000
    lr: float = 0.001
    lr_min: float = 1e-5
    stress_weight: float = 100.0
    patience: int = 500
    load_ratios: List[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    )

    # ── 物理制約 ──────────────────────────────────────────
    linear_scaling: bool = True

    # ── データキー（None = 自動検出）──────────────────────
    disp_key: Optional[str] = None
    stress_key: Optional[str] = None

    # ------------------------------------------------------------------
    # 自動キャリブレーション
    # ------------------------------------------------------------------
    def auto_calibrate(self, mesh, load_N: Optional[float] = None) -> None:
        """メッシュの統計量から正規化定数を自動計算する。"""
        pos = mesh.points
        extent = float(max(pos.max(axis=0) - pos.min(axis=0)))

        if self.norm_coord is None:
            self.norm_coord = extent
        if load_N is not None:
            self.train_load = load_N

        keys = list(mesh.point_data.keys())
        if self.disp_key is None:
            self.disp_key = self._find_key(
                keys, [["displacement", "disp"], ["u_", "uvw"]]
            )
        if self.stress_key is None:
            self.stress_key = self._find_key(
                keys, [["mises", "von"], ["eqv", "equivalent"], ["stress"]]
            )

        if self.disp_key and self.norm_disp is None:
            d = mesh.point_data[self.disp_key]
            self.norm_disp = max(float(np.max(np.abs(d)) * self.to_mm), 1e-12)
        if self.stress_key and self.norm_stress is None:
            s = mesh.point_data[self.stress_key]
            if s.ndim > 1:
                s = np.linalg.norm(s, axis=1)
            self.norm_stress = max(float(np.max(np.abs(s)) * self.to_mpa), 1e-12)

    # ------------------------------------------------------------------
    # キー探索
    # ------------------------------------------------------------------
    @staticmethod
    def _find_key(keys: list, priority_groups: list) -> Optional[str]:
        for group in priority_groups:
            for k in keys:
                if any(s in k.lower() for s in group):
                    return k
        return None

    # ------------------------------------------------------------------
    # 永続化
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """設定を JSON に保存する。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=str, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "GNNConfig":
        """JSON から設定を読み込む。"""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(**d)

    # ------------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            f"  座標正規化  : {self.norm_coord}",
            f"  変位キー    : {self.disp_key}  (正規化 = {self.norm_disp} mm)",
            f"  応力キー    : {self.stress_key}  (正規化 = {self.norm_stress} MPa)",
            f"  基準荷重    : {self.train_load} N",
            f"  モデル      : {self.n_layers}層 × hidden={self.hidden_dim}",
            f"  学習        : epochs={self.epochs}, lr={self.lr}, "
            f"patience={self.patience}",
        ]
        return "\n".join(lines)

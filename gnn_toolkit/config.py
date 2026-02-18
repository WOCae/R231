"""
GNNConfig — 設定クラス

データから正規化定数を自動計算し、JSON で永続化する。
コンストラクタ引数で任意のパラメータをオーバーライド可能。
v3: ジオメトリ特徴量 (include_geometry) 追加でメッシュ形状変更に対応。
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
    n_features: int = 14      # ジオメトリ特徴量 (9) + is_fixed + is_load + Fx + Fy + Fz
    n_outputs: int = 4        # ux, uy, uz, stress

    # ── ジオメトリ特徴量 ──────────────────────────────────
    include_geometry: bool = True   # True: 12次元入力, False: 5次元入力（レガシー）

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
    # ── 境界条件パターン（学習時に検出・保存、推論時に再適用）──
    bc_fixed_axis: Optional[int] = None       # 拘束面の軸 (0=X, 1=Y, 2=Z)
    bc_fixed_side: Optional[str] = None       # "min" or "max"
    bc_load_axis: Optional[int] = None        # 荷重面の軸
    bc_load_side: Optional[str] = None        # "min" or "max"
    bc_load_direction: Optional[List[float]] = None  # 荷重方向単位ベクトル [dx, dy, dz]
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

    def auto_calibrate_multi(self, meshes: list, load_values: list) -> None:
        """複数メッシュからの正規化定数を自動計算する。"""
        self.auto_calibrate(meshes[0], load_values[0])
        # 他のメッシュの最大値で更新
        for mesh in meshes[1:]:
            keys = list(mesh.point_data.keys())
            dk = self.disp_key or self._find_key(
                keys, [["displacement", "disp"], ["u_", "uvw"]]
            )
            sk = self.stress_key or self._find_key(
                keys, [["mises", "von"], ["eqv", "equivalent"], ["stress"]]
            )
            if dk and dk in mesh.point_data:
                d = mesh.point_data[dk]
                v = max(float(np.max(np.abs(d)) * self.to_mm), 1e-12)
                if self.norm_disp is not None:
                    self.norm_disp = max(self.norm_disp, v)
            if sk and sk in mesh.point_data:
                s = mesh.point_data[sk]
                if s.ndim > 1:
                    s = np.linalg.norm(s, axis=1)
                v = max(float(np.max(np.abs(s)) * self.to_mpa), 1e-12)
                if self.norm_stress is not None:
                    self.norm_stress = max(self.norm_stress, v)

    # ------------------------------------------------------------------
    # n_features を自動設定
    # ------------------------------------------------------------------
    def update_n_features(self) -> None:
        """include_geometry に応じて n_features を設定する。"""
        self.n_features = 14 if self.include_geometry else 8

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
        # v2 → v3 後方互換
        if "include_geometry" not in d:
            d["include_geometry"] = False
            d["n_features"] = 5
        # v3.0 (12次元) → v3.1 (14次元) 互換: n_features はロード後に再設定
        return cls(**d)

    # ------------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------------
    def summary(self) -> str:
        mode = f"ジオメトリ{self.n_features}次元" if self.include_geometry else f"レガシー{self.n_features}次元"
        axis_lbl = {0: "X", 1: "Y", 2: "Z"}
        bc_info = "未検出"
        if self.bc_fixed_axis is not None:
            bc_info = (f"拘束={axis_lbl.get(self.bc_fixed_axis,'?')}-{self.bc_fixed_side}, "
                       f"荷重={axis_lbl.get(self.bc_load_axis,'?')}-{self.bc_load_side}, "
                       f"方向={self.bc_load_direction}")
        lines = [
            f"  座標正規化  : {self.norm_coord}",
            f"  変位キー    : {self.disp_key}  (正規化 = {self.norm_disp} mm)",
            f"  応力キー    : {self.stress_key}  (正規化 = {self.norm_stress} MPa)",
            f"  基準荷重    : {self.train_load} N",
            f"  境界条件  : {bc_info}",
            f"  入力特徴量  : {mode} (n_features={self.n_features})",
            f"  モデル      : {self.n_layers}層 × hidden={self.hidden_dim}",
            f"  学習        : epochs={self.epochs}, lr={self.lr}, "
            f"patience={self.patience}",
        ]
        return "\n".join(lines)

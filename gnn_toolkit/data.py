"""
FEADataProcessor — VTU 自動解析・境界条件検出・グラフ変換

VTU ファイルを読み込み、PyTorch Geometric の Data オブジェクトに変換する。
境界条件（拘束面 / 荷重面）は変位データから自動検出する。
v3.1: 荷重方向ベクトル化で引張・曲げ等の複数荷重方向に対応。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import torch
from torch_geometric.data import Data

from .config import GNNConfig


class FEADataProcessor:
    """VTU ファイルの自動解析・PyG Data 変換を行うユーティリティ。"""

    # ------------------------------------------------------------------
    # VTU 解析
    # ------------------------------------------------------------------
    @staticmethod
    def analyze(path: str) -> pv.UnstructuredGrid:
        """VTU の中身を解析して表示し、point_data 変換済みメッシュを返す。"""
        mesh = pv.read(path)
        info: Dict[str, object] = {
            "ノード": mesh.n_points,
            "要素":   mesh.n_cells,
            "範囲":   mesh.bounds,
            "Point":  list(mesh.point_data.keys()),
            "Cell":   list(mesh.cell_data.keys()),
        }
        print(f"=== {path} ===")
        for k, v in info.items():
            print(f"  {k}: {v}")

        mesh_p = mesh.cell_data_to_point_data()
        extra = set(mesh_p.point_data.keys()) - set(mesh.point_data.keys())
        if extra:
            print(f"  Cell→Point 変換で追加: {list(extra)}")
        return mesh_p

    # ------------------------------------------------------------------
    # 境界条件の自動検出
    # ------------------------------------------------------------------
    @staticmethod
    def detect_bc(
        mesh: pv.UnstructuredGrid,
        disp_data: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        境界条件の自動検出。

        Parameters
        ----------
        mesh : pv.UnstructuredGrid
            解析済みメッシュ
        disp_data : np.ndarray, optional
            変位データ（あれば高精度）

        Returns
        -------
        is_fixed : ndarray[float]  拘束ノード (1.0 / 0.0)
        is_load  : ndarray[float]  荷重ノード (1.0 / 0.0)
        """
        pos = mesh.points
        n = pos.shape[0]

        if disp_data is not None and disp_data.ndim >= 1:
            mag = (
                np.linalg.norm(disp_data, axis=1)
                if disp_data.ndim > 1
                else np.abs(disp_data)
            )
            threshold = np.max(mag) * 1e-6
            is_fixed = (mag <= threshold).astype(float)

            ranges = pos.max(axis=0) - pos.min(axis=0)
            max_idx = int(np.argmax(mag))
            is_load = np.zeros(n, dtype=float)
            for ax in range(3):
                if ranges[ax] < 1e-10:
                    continue
                tol = ranges[ax] * 0.01
                if abs(pos[max_idx, ax] - pos[:, ax].max()) < tol:
                    is_load = (pos[:, ax] >= pos[:, ax].max() - tol).astype(float)
                    break
                if abs(pos[max_idx, ax] - pos[:, ax].min()) < tol:
                    is_load = (pos[:, ax] <= pos[:, ax].min() + tol).astype(float)
                    break
        else:
            # フォールバック: 最長軸の両端
            ranges = pos.max(axis=0) - pos.min(axis=0)
            ax = int(np.argmax(ranges))
            tol = ranges[ax] * 0.01
            is_fixed = (pos[:, ax] <= pos[:, ax].min() + tol).astype(float)
            is_load  = (pos[:, ax] >= pos[:, ax].max() - tol).astype(float)

        return is_fixed, is_load

    # ------------------------------------------------------------------
    # 境界条件パターンの検出・保存・再適用
    # ------------------------------------------------------------------
    @staticmethod
    def detect_bc_pattern(
        mesh: pv.UnstructuredGrid,
        disp_data: np.ndarray,
    ) -> dict:
        """
        変位データから境界条件パターン（拘束面・荷重面の軸と側）を検出する。

        Returns
        -------
        dict  — bc_fixed_axis, bc_fixed_side, bc_load_axis, bc_load_side
        """
        pos = mesh.points
        mag = np.linalg.norm(disp_data, axis=1) if disp_data.ndim > 1 else np.abs(disp_data)
        threshold = np.max(mag) * 1e-6

        # 拘束面: 変位≈ 0 のノードが集中する軸・側を特定
        fixed_mask = mag <= threshold
        fixed_pos = pos[fixed_mask]
        ranges = pos.max(axis=0) - pos.min(axis=0)

        bc_fixed_axis, bc_fixed_side = 0, "min"
        if len(fixed_pos) > 0:
            for ax in range(3):
                if ranges[ax] < 1e-10:
                    continue
                tol = ranges[ax] * 0.02
                at_min = np.sum(np.abs(fixed_pos[:, ax] - pos[:, ax].min()) < tol)
                at_max = np.sum(np.abs(fixed_pos[:, ax] - pos[:, ax].max()) < tol)
                if at_min > len(fixed_pos) * 0.5:
                    bc_fixed_axis, bc_fixed_side = ax, "min"
                    break
                if at_max > len(fixed_pos) * 0.5:
                    bc_fixed_axis, bc_fixed_side = ax, "max"
                    break

        # 荷重面: 変位最大ノードが属する面
        max_idx = int(np.argmax(mag))
        bc_load_axis, bc_load_side = 0, "max"
        for ax in range(3):
            if ranges[ax] < 1e-10:
                continue
            tol = ranges[ax] * 0.01
            if abs(pos[max_idx, ax] - pos[:, ax].max()) < tol:
                bc_load_axis, bc_load_side = ax, "max"
                break
            if abs(pos[max_idx, ax] - pos[:, ax].min()) < tol:
                bc_load_axis, bc_load_side = ax, "min"
                break

        return {
            "bc_fixed_axis": bc_fixed_axis,
            "bc_fixed_side": bc_fixed_side,
            "bc_load_axis": bc_load_axis,
            "bc_load_side": bc_load_side,
        }

    @staticmethod
    def apply_bc_pattern(
        pos: np.ndarray,
        config: "GNNConfig",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        保存済みのBCパターンを任意のメッシュ座標に適用する。

        学習時に検出した「どの軸のどの側が拘束/荷重」を、
        推論対象メッシュの座標に再適用する。
        """
        n = pos.shape[0]
        ranges = pos.max(axis=0) - pos.min(axis=0)

        def _face_mask(axis: int, side: str) -> np.ndarray:
            if ranges[axis] < 1e-10:
                return np.zeros(n, dtype=float)
            tol = ranges[axis] * 0.01
            if side == "min":
                return (pos[:, axis] <= pos[:, axis].min() + tol).astype(float)
            else:
                return (pos[:, axis] >= pos[:, axis].max() - tol).astype(float)

        is_fixed = _face_mask(config.bc_fixed_axis, config.bc_fixed_side)
        is_load = _face_mask(config.bc_load_axis, config.bc_load_side)
        return is_fixed, is_load

    # ------------------------------------------------------------------
    # ジオメトリ特徴量の計算
    # ------------------------------------------------------------------
    @staticmethod
    def compute_geometry_features(
        pos: np.ndarray,
        edge_index: torch.Tensor,
    ) -> np.ndarray:
        """
        ノードごとのジオメトリ特徴量を計算する。

        Parameters
        ----------
        pos : ndarray (N, 3)
            ノード座標
        edge_index : Tensor (2, E)
            エッジインデックス（双方向）

        Returns
        -------
        ndarray (N, 9)
            [正規化座標(3), 正規化次数(1), 平均エッジ長(1),
             エッジ長標準偏差(1), 擬似法線(3)]
        """
        n = len(pos)

        # 正規化座標 (bbox → [0, 1])
        bbox_min = pos.min(axis=0)
        bbox_range = pos.max(axis=0) - bbox_min
        bbox_range[bbox_range < 1e-12] = 1.0
        norm_coords = (pos - bbox_min) / bbox_range

        # ベクトル化された隣接情報計算
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()

        # 次数 (各ノードの隣接数)
        degree = np.bincount(src, minlength=n).astype(np.float32)

        # エッジ長 (各エッジの距離)
        diff_all = pos[dst] - pos[src]  # (E, 3)
        dists_all = np.linalg.norm(diff_all, axis=1)  # (E,)

        # ノードごとの平均エッジ長・標準偏差・擬似法線をベクトル化
        edge_len_mean = np.zeros(n, dtype=np.float32)
        edge_len_std = np.zeros(n, dtype=np.float32)
        pseudo_normal = np.zeros((n, 3), dtype=np.float32)

        # scatter: ノードごとの合計
        edge_len_sum = np.bincount(src, weights=dists_all, minlength=n)
        mask = degree > 0
        edge_len_mean[mask] = (edge_len_sum[mask] / degree[mask]).astype(np.float32)

        # 方向ベクトルの合計 → 擬似法線
        for ax in range(3):
            pseudo_normal[:, ax] = np.bincount(
                src, weights=diff_all[:, ax], minlength=n
            ).astype(np.float32)
        pseudo_normal[mask] /= degree[mask, None]
        norms = np.linalg.norm(pseudo_normal, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pseudo_normal /= norms

        # 標準偏差（scatter で二乗平均 - 平均二乗）
        sq_sum = np.bincount(src, weights=dists_all ** 2, minlength=n)
        multi_mask = degree > 1
        variance = np.zeros(n, dtype=np.float32)
        variance[multi_mask] = (
            sq_sum[multi_mask] / degree[multi_mask] - edge_len_mean[multi_mask] ** 2
        )
        variance = np.maximum(variance, 0.0)  # 数値誤差対策
        edge_len_std[multi_mask] = np.sqrt(variance[multi_mask])

        # 正規化 (最大値で割る、ゼロ除算防止)
        def _safe_normalize(arr: np.ndarray) -> np.ndarray:
            m = arr.max()
            return arr / m if m > 0 else arr

        return np.column_stack([
            norm_coords,                              # 3
            _safe_normalize(degree)[:, None],          # 1
            _safe_normalize(edge_len_mean)[:, None],   # 1
            _safe_normalize(edge_len_std)[:, None],    # 1
            pseudo_normal,                             # 3
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # 荷重方向の自動検出
    # ------------------------------------------------------------------
    @staticmethod
    def detect_load_direction(
        mesh: pv.UnstructuredGrid,
        disp_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        変位データから荷重方向の単位ベクトルを推定する。

        変位が最大のノードの変位方向を荷重方向とみなす。
        変位データがない場合は最長軸方向 [1,0,0] を返す。

        Returns
        -------
        ndarray (3,) — 単位荷重方向ベクトル
        """
        if disp_data is not None and disp_data.ndim == 2 and disp_data.shape[1] == 3:
            mag = np.linalg.norm(disp_data, axis=1)
            max_idx = int(np.argmax(mag))
            if mag[max_idx] > 1e-12:
                d = disp_data[max_idx]
                return d / np.linalg.norm(d)
        # フォールバック: 最長軸方向
        ranges = mesh.points.max(axis=0) - mesh.points.min(axis=0)
        ax = int(np.argmax(ranges))
        direction = np.zeros(3)
        direction[ax] = 1.0
        return direction

    # ------------------------------------------------------------------
    # VTU → PyG Data
    # ------------------------------------------------------------------
    @staticmethod
    def to_graph(
        vtu_path: str,
        load_N: float,
        config: GNNConfig,
        load_direction: Optional[np.ndarray] = None,
        predict_mode: bool = False,
    ) -> Data:
        """
        VTU ファイルを PyTorch Geometric の Data オブジェクトに変換する。

        Parameters
        ----------
        load_direction : ndarray (3,), optional
            荷重方向の単位ベクトル。None の場合は変位から自動検出。
        predict_mode : bool
            True の場合、VTUのFEA結果を無視し、configに保存済みの
            BCパターンを適用する。推論時に使用。
        """
        mesh = pv.read(vtu_path)
        mesh = mesh.cell_data_to_point_data()
        pos = mesh.points

        if predict_mode and config.bc_fixed_axis is not None:
            # --- 推論モード: FEA結果を使わず、保存済みBCパターンを適用 ---
            y = np.zeros((pos.shape[0], config.n_outputs))
            is_fixed, is_load = FEADataProcessor.apply_bc_pattern(pos, config)
            if load_direction is None and config.bc_load_direction is not None:
                load_direction = np.array(config.bc_load_direction)
            elif load_direction is None:
                load_direction = FEADataProcessor.detect_load_direction(mesh, None)
        else:
            # --- 学習モード: FEA結果からBCを検出 ---
            y = np.zeros((pos.shape[0], config.n_outputs))
            disp_data = None
            if config.disp_key and config.disp_key in mesh.point_data:
                disp_data = mesh.point_data[config.disp_key]
                y[:, :3] = (disp_data * config.to_mm) / config.norm_disp
            if config.stress_key and config.stress_key in mesh.point_data:
                s = mesh.point_data[config.stress_key]
                if s.ndim > 1:
                    s = np.linalg.norm(s, axis=1)
                y[:, 3] = (s * config.to_mpa) / config.norm_stress

            is_fixed, is_load = FEADataProcessor.detect_bc(mesh, disp_data)
            if load_direction is None:
                load_direction = FEADataProcessor.detect_load_direction(mesh, disp_data)

        # エッジ（双方向）
        edges = mesh.extract_all_edges().lines.reshape(-1, 3)[:, 1:]
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        ei = torch.cat([ei, ei.flip(0)], dim=1)

        # 特徴量構築
        x = FEADataProcessor._build_features(
            pos, ei, is_fixed, is_load, load_N, load_direction, config
        )
        return Data(x=x, edge_index=ei, y=torch.tensor(y, dtype=torch.float))

    # ------------------------------------------------------------------
    # INP → PyG Data
    # ------------------------------------------------------------------
    @staticmethod
    def to_graph_from_inp(
        inp_path: str,
        load_N: Optional[float],
        config: GNNConfig,
        load_direction: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        CalculiX .inp ファイルから PyG Data を構築する。

        Parameters
        ----------
        inp_path : str
            CalculiX .inp ファイルパス
        load_N : float, optional
            荷重値 [N]。None の場合はファイル内 CLOAD から自動計算
        config : GNNConfig
            設定
        load_direction : ndarray (3,), optional
            荷重方向の単位ベクトル。None の場合は CLOAD から自動取得

        Returns
        -------
        (Data, InpFileReader)
            PyG Data と読み取り済みリーダー
        """
        from .inp_reader import InpFileReader

        reader = InpFileReader().read(inp_path)
        pos = reader.get_positions()

        if load_N is None:
            load_N = reader.get_total_load()

        # エッジ（双方向）
        edges = reader.get_edges()
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        ei = torch.cat([ei, ei.flip(0)], dim=1)

        # 拘束・荷重マスク（.inp で明示定義）
        is_fixed = reader.get_fixed_mask()
        is_load = reader.get_load_mask()

        # 荷重方向: CLOAD から自動取得
        if load_direction is None:
            load_direction = reader.get_load_direction()

        # 特徴量構築
        x = FEADataProcessor._build_features(
            pos, ei, is_fixed, is_load, load_N, load_direction, config
        )
        y = torch.zeros((len(pos), config.n_outputs), dtype=torch.float)

        return Data(x=x, edge_index=ei, y=y), reader

    # ------------------------------------------------------------------
    # 特徴量構築（共通）
    # ------------------------------------------------------------------
    @staticmethod
    def _build_features(
        pos: np.ndarray,
        edge_index: torch.Tensor,
        is_fixed: np.ndarray,
        is_load: np.ndarray,
        load_N: float,
        load_direction: np.ndarray,
        config: GNNConfig,
    ) -> torch.Tensor:
        """
        ノード座標・BC マスクから入力特徴量テンソルを構築する。

        include_geometry=True の場合はジオメトリ 14 次元、
        False の場合はレガシー 8 次元を返す。
        """
        load_ratio = load_N / config.train_load
        fx = is_load * load_direction[0] * load_ratio
        fy = is_load * load_direction[1] * load_ratio
        fz = is_load * load_direction[2] * load_ratio

        if config.include_geometry:
            geo = FEADataProcessor.compute_geometry_features(pos, edge_index)
            cols = [geo, is_fixed, is_load, fx, fy, fz]
        else:
            pos_norm = pos / config.norm_coord
            cols = [pos_norm, is_fixed, is_load, fx, fy, fz]

        return torch.tensor(np.column_stack(cols), dtype=torch.float)

    # ------------------------------------------------------------------
    # VTU ファイル一覧取得
    # ------------------------------------------------------------------
    @staticmethod
    def list_vtu_files(directory: str = ".") -> List[str]:
        """指定ディレクトリ内の .vtu ファイル一覧を返す。"""
        import glob
        import os

        pattern = os.path.join(directory, "*.vtu")
        return sorted(glob.glob(pattern))

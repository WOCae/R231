"""
FEADataProcessor — VTU 自動解析・境界条件検出・グラフ変換

VTU ファイルを読み込み、PyTorch Geometric の Data オブジェクトに変換する。
境界条件（拘束面 / 荷重面）は変位データから自動検出する。
v3: ジオメトリ特徴量 (正規化座標・次数・エッジ長統計・擬似法線) 追加。
"""

from __future__ import annotations

from collections import defaultdict
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
        bbox_max = pos.max(axis=0)
        bbox_range = bbox_max - bbox_min
        bbox_range[bbox_range < 1e-12] = 1.0
        norm_coords = (pos - bbox_min) / bbox_range

        # 隣接情報構築
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()

        neighbors = defaultdict(list)
        for s, d in zip(src, dst):
            neighbors[int(s)].append(int(d))

        degree = np.zeros(n, dtype=np.float32)
        edge_len_mean = np.zeros(n, dtype=np.float32)
        edge_len_std = np.zeros(n, dtype=np.float32)
        pseudo_normal = np.zeros((n, 3), dtype=np.float32)

        for i in range(n):
            nbrs = neighbors[i]
            deg = len(nbrs)
            degree[i] = deg
            if deg == 0:
                continue
            nbr_pts = pos[nbrs]
            diff = nbr_pts - pos[i]
            dists = np.linalg.norm(diff, axis=1)
            edge_len_mean[i] = dists.mean()
            edge_len_std[i] = dists.std() if deg > 1 else 0.0
            mean_dir = diff.mean(axis=0)
            norm = np.linalg.norm(mean_dir)
            if norm > 1e-12:
                pseudo_normal[i] = mean_dir / norm

        # 正規化
        max_deg = degree.max() if degree.max() > 0 else 1.0
        degree_norm = degree / max_deg

        max_el = edge_len_mean.max() if edge_len_mean.max() > 0 else 1.0
        edge_len_mean_norm = edge_len_mean / max_el

        max_es = edge_len_std.max() if edge_len_std.max() > 0 else 1.0
        edge_len_std_norm = edge_len_std / max_es

        return np.column_stack([
            norm_coords,                   # 3
            degree_norm[:, None],          # 1
            edge_len_mean_norm[:, None],   # 1
            edge_len_std_norm[:, None],    # 1
            pseudo_normal,                 # 3
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # VTU → PyG Data
    # ------------------------------------------------------------------
    @staticmethod
    def to_graph(vtu_path: str, load_N: float, config: GNNConfig) -> Data:
        """VTU ファイルを PyTorch Geometric の Data オブジェクトに変換する。"""
        mesh = pv.read(vtu_path)
        mesh = mesh.cell_data_to_point_data()
        pos = mesh.points

        # ラベル
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

        # 境界条件
        is_fixed, is_load = FEADataProcessor.detect_bc(mesh, disp_data)

        # エッジ（双方向）
        edges = mesh.extract_all_edges().lines.reshape(-1, 3)[:, 1:]
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        ei = torch.cat([ei, ei.flip(0)], dim=1)

        # 特徴量構築
        if config.include_geometry:
            # ジオメトリ12次元: geo(9) + is_fixed(1) + is_load(1) + load_ratio(1)
            geo = FEADataProcessor.compute_geometry_features(pos, ei)
            load_ratio = is_load * (load_N / config.train_load)
            x = torch.tensor(
                np.column_stack([geo, is_fixed, is_load, load_ratio]),
                dtype=torch.float,
            )
        else:
            # レガシー5次元: pos_norm(3) + is_fixed(1) + load_feat(1)
            pos_norm = pos / config.norm_coord
            load_feat = is_load * (load_N / config.train_load)
            x = torch.tensor(
                np.column_stack([pos_norm, is_fixed, load_feat]),
                dtype=torch.float,
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

        # 特徴量構築
        if config.include_geometry:
            geo = FEADataProcessor.compute_geometry_features(pos, ei)
            load_ratio = is_load * (load_N / config.train_load)
            x = torch.tensor(
                np.column_stack([geo, is_fixed, is_load, load_ratio]),
                dtype=torch.float,
            )
        else:
            pos_norm = pos / config.norm_coord
            load_feat = is_load * (load_N / config.train_load)
            x = torch.tensor(
                np.column_stack([pos_norm, is_fixed, load_feat]),
                dtype=torch.float,
            )

        # ラベルなし（推論専用）
        y = torch.zeros((len(pos), config.n_outputs), dtype=torch.float)

        return Data(x=x, edge_index=ei, y=y), reader

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

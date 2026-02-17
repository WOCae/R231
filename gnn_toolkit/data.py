"""
FEADataProcessor — VTU 自動解析・境界条件検出・グラフ変換

VTU ファイルを読み込み、PyTorch Geometric の Data オブジェクトに変換する。
境界条件（拘束面 / 荷重面）は変位データから自動検出する。
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
    # VTU → PyG Data
    # ------------------------------------------------------------------
    @staticmethod
    def to_graph(vtu_path: str, load_N: float, config: GNNConfig) -> Data:
        """VTU ファイルを PyTorch Geometric の Data オブジェクトに変換する。"""
        mesh = pv.read(vtu_path)
        mesh = mesh.cell_data_to_point_data()
        pos = mesh.points
        pos_norm = pos / config.norm_coord

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
        load_feat = is_load * (load_N / config.train_load)

        x = torch.tensor(
            np.column_stack([pos_norm, is_fixed, load_feat]),
            dtype=torch.float,
        )

        # エッジ（双方向）
        edges = mesh.extract_all_edges().lines.reshape(-1, 3)[:, 1:]
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        ei = torch.cat([ei, ei.flip(0)], dim=1)

        return Data(x=x, edge_index=ei, y=torch.tensor(y, dtype=torch.float))

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

"""
InpFileReader — CalculiX .inp ファイルパーサー

FreeCAD / CalculiX の入力ファイルを解析し、メッシュ・境界条件・荷重情報を取得する。
推論時に明示的な拘束・荷重条件を指定することで、自動検出の不確実性を排除する。

対応要素タイプ: C3D4 (4節点四面体), C3D10 (10節点四面体), C3D8 (8節点六面体)
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# =====================================================================
# 要素タイプごとのエッジ定義（要素内ローカル 0-based インデックス）
# =====================================================================

# C3D4 — 4節点四面体: 6辺
_C3D4_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# C3D10 — 10節点四面体: 12辺（中間節点経由）
#   Corners: 0-3,  Midside: 4(0-1), 5(1-2), 6(0-2), 7(0-3), 8(1-3), 9(2-3)
_C3D10_EDGES = [
    (0, 4), (4, 1),   # 辺 0-1
    (1, 5), (5, 2),   # 辺 1-2
    (2, 6), (6, 0),   # 辺 2-0
    (0, 7), (7, 3),   # 辺 0-3
    (1, 8), (8, 3),   # 辺 1-3
    (2, 9), (9, 3),   # 辺 2-3
]

# C3D8 — 8節点六面体: 12辺
_C3D8_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 上面
    (0, 4), (1, 5), (2, 6), (3, 7),  # 垂直辺
]

_ELEMENT_EDGES: Dict[str, list] = {
    "C3D4": _C3D4_EDGES,
    "C3D10": _C3D10_EDGES,
    "C3D8": _C3D8_EDGES,
}

# VTK セルタイプ（PyVista メッシュ構築用）
_VTK_CELL_TYPES: Dict[str, int] = {
    "C3D4": 10,   # VTK_TETRA
    "C3D10": 24,  # VTK_QUADRATIC_TETRA
    "C3D8": 12,   # VTK_HEXAHEDRON
}


# =====================================================================
# パーサー本体
# =====================================================================
class InpFileReader:
    """
    CalculiX .inp ファイルを解析するパーサー。

    FreeCAD が出力する CalculiX 入力ファイルに対応し、以下の情報を抽出する:
    - ノード座標 (*NODE)
    - 要素接続 (*ELEMENT)
    - ノード集合 (*NSET — 通常 / GENERATE)
    - 境界条件 (*BOUNDARY)
    - 集中荷重 (*CLOAD)

    使い方::

        reader = InpFileReader().read("FEMMeshNetgen.inp")
        print(reader.summary())
        pos = reader.get_positions()          # (N, 3) 座標
        is_fixed = reader.get_fixed_mask()    # (N,) 拘束マスク
        is_load = reader.get_load_mask()      # (N,) 荷重マスク
        total_load = reader.get_total_load()  # 合力 [N]
    """

    def __init__(self) -> None:
        self.nodes: Dict[int, Tuple[float, float, float]] = {}
        self.elements: Dict[int, List[int]] = {}
        self.element_type: Optional[str] = None
        self.nsets: Dict[str, Set[int]] = defaultdict(set)
        self._boundary_nodes: Set[int] = set()
        self._load_entries: List[Tuple[int, int, float]] = []
        self._node_id_to_idx: Dict[int, int] = {}
        self._idx_to_node_id: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # ファイル読み込み
    # ------------------------------------------------------------------
    def read(self, path: str) -> "InpFileReader":
        """
        .inp ファイルを読み込む。

        Parameters
        ----------
        path : str
            CalculiX .inp ファイルパス

        Returns
        -------
        self
        """
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        state: Optional[str] = None
        current_nset: Optional[str] = None
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            i += 1

            # コメント行
            if line.startswith("**"):
                continue

            # キーワード行
            if line.startswith("*"):
                kw = line.upper()

                if kw.startswith("*NODE"):
                    state = "NODE"
                    continue
                elif kw.startswith("*ELEMENT"):
                    state = "ELEMENT"
                    m = re.search(r"TYPE\s*=\s*(\w+)", kw)
                    if m:
                        self.element_type = m.group(1).upper()
                    continue
                elif kw.startswith("*NSET"):
                    m = re.search(r"NSET\s*=\s*(\w+)", kw)
                    if m:
                        current_nset = m.group(1).upper()
                    if "GENERATE" in kw:
                        state = "NSET_GENERATE"
                    else:
                        state = "NSET"
                    continue
                elif kw.startswith("*BOUNDARY"):
                    state = "BOUNDARY"
                    continue
                elif kw.startswith("*CLOAD"):
                    state = "CLOAD"
                    continue
                else:
                    state = None
                    continue

            # 空行スキップ
            if not line:
                continue

            # ── データ行の解析 ────────────────────────
            if state == "NODE":
                parts = line.split(",")
                if len(parts) >= 4:
                    nid = int(parts[0].strip())
                    x = float(parts[1].strip())
                    y = float(parts[2].strip())
                    z = float(parts[3].strip())
                    self.nodes[nid] = (x, y, z)

            elif state == "ELEMENT":
                # 継続行対応（末尾カンマで次行に続く）
                full_line = line
                while full_line.rstrip().endswith(",") and i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith("*"):
                        break
                    i += 1
                    full_line += " " + next_line
                parts = [p.strip() for p in full_line.split(",") if p.strip()]
                if parts:
                    eid = int(parts[0])
                    node_ids = [int(p) for p in parts[1:]]
                    self.elements[eid] = node_ids

            elif state == "NSET":
                parts = [p.strip() for p in line.split(",") if p.strip()]
                for p in parts:
                    try:
                        self.nsets[current_nset].add(int(p))
                    except ValueError:
                        pass

            elif state == "NSET_GENERATE":
                parts = [int(p.strip()) for p in line.split(",") if p.strip()]
                if len(parts) >= 2:
                    start, end = parts[0], parts[1]
                    step = parts[2] if len(parts) >= 3 else 1
                    for nid in range(start, end + 1, step):
                        self.nsets[current_nset].add(nid)
                state = None  # GENERATE は単行

            elif state == "BOUNDARY":
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) >= 3:
                    resolved = self._resolve_nset_or_node(parts[0])
                    self._boundary_nodes.update(resolved)

            elif state == "CLOAD":
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) >= 3:
                    dof = int(parts[1])
                    value = float(parts[2])
                    resolved = self._resolve_nset_or_node(parts[0])
                    for nid in resolved:
                        self._load_entries.append((nid, dof, value))

        # ノードID → 連続インデックス マッピング
        sorted_ids = sorted(self.nodes.keys())
        self._node_id_to_idx = {nid: idx for idx, nid in enumerate(sorted_ids)}
        self._idx_to_node_id = {idx: nid for nid, idx in self._node_id_to_idx.items()}

        return self

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------
    def _resolve_nset_or_node(self, name: str) -> Set[int]:
        """NSET 名ならその中身を、整数ならそのノードを返す。"""
        try:
            return {int(name)}
        except ValueError:
            key = name.upper()
            return self.nsets.get(key, set())

    # ------------------------------------------------------------------
    # プロパティ
    # ------------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    # ------------------------------------------------------------------
    # データ取得
    # ------------------------------------------------------------------
    def get_positions(self) -> np.ndarray:
        """ノード座標を (N, 3) の ndarray で返す（ID昇順）。"""
        sorted_ids = sorted(self.nodes.keys())
        return np.array([self.nodes[nid] for nid in sorted_ids], dtype=np.float64)

    def get_edges(self) -> np.ndarray:
        """
        要素トポロジーからエッジペアを抽出する。

        Returns
        -------
        ndarray (E, 2) — 0-based index のエッジペア（片方向）
        """
        if self.element_type not in _ELEMENT_EDGES:
            raise ValueError(
                f"未対応の要素タイプ: {self.element_type}. "
                f"対応: {list(_ELEMENT_EDGES.keys())}"
            )

        edge_template = _ELEMENT_EDGES[self.element_type]
        edge_set: Set[Tuple[int, int]] = set()

        for node_ids in self.elements.values():
            for li, lj in edge_template:
                ni = self._node_id_to_idx[node_ids[li]]
                nj = self._node_id_to_idx[node_ids[lj]]
                edge = (min(ni, nj), max(ni, nj))
                edge_set.add(edge)

        return np.array(sorted(edge_set), dtype=np.int64)

    def get_fixed_mask(self) -> np.ndarray:
        """拘束ノードの mask (N,) を返す。1.0=拘束, 0.0=自由。"""
        n = self.n_nodes
        fixed = np.zeros(n, dtype=np.float32)
        for nid in self._boundary_nodes:
            if nid in self._node_id_to_idx:
                fixed[self._node_id_to_idx[nid]] = 1.0
        return fixed

    def get_load_mask(self) -> np.ndarray:
        """荷重ノードの mask (N,) を返す。1.0=荷重, 0.0=非荷重。"""
        n = self.n_nodes
        load = np.zeros(n, dtype=np.float32)
        for nid, _, _ in self._load_entries:
            if nid in self._node_id_to_idx:
                load[self._node_id_to_idx[nid]] = 1.0
        return load

    def get_total_load(self) -> float:
        """CLOAD の合力ベクトルのノルムを返す [N]。"""
        force = np.zeros(3)
        for _, dof, value in self._load_entries:
            if 1 <= dof <= 3:
                force[dof - 1] += value
        return float(np.linalg.norm(force))

    def get_load_direction(self) -> np.ndarray:
        """
        CLOAD から荷重方向の単位ベクトルを返す。

        Returns
        -------
        ndarray (3,) — 荷重方向の単位ベクトル
        """
        force = np.zeros(3)
        for _, dof, value in self._load_entries:
            if 1 <= dof <= 3:
                force[dof - 1] += value
        norm = np.linalg.norm(force)
        if norm > 1e-12:
            return force / norm
        # フォールバック: Y方向
        return np.array([0.0, 1.0, 0.0])

    # ------------------------------------------------------------------
    # サマリー
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """解析結果のサマリー文字列。"""
        fixed_count = int(self.get_fixed_mask().sum())
        load_count = int(self.get_load_mask().sum())

        force = np.zeros(3)
        for _, dof, value in self._load_entries:
            if 1 <= dof <= 3:
                force[dof - 1] += value
        total = float(np.linalg.norm(force))

        axis_labels = ["X", "Y", "Z"]
        force_parts = [
            f"{ax}={f:.1f}N" for ax, f in zip(axis_labels, force) if abs(f) > 1e-6
        ]
        force_str = ", ".join(force_parts) if force_parts else "なし"

        # 形状情報
        pos = self.get_positions()
        bbox_min = pos.min(axis=0)
        bbox_max = pos.max(axis=0)
        size = bbox_max - bbox_min

        lines = [
            f"  ノード数    : {self.n_nodes}",
            f"  要素数      : {self.n_elements} ({self.element_type})",
            f"  エッジ数    : {len(self.get_edges())}",
            f"  形状サイズ  : {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f}",
            f"  拘束ノード  : {fixed_count}",
            f"  荷重ノード  : {load_count}",
            f"  合計荷重    : {total:.1f} N ({force_str})",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # PyVista メッシュ構築
    # ------------------------------------------------------------------
    def to_pyvista(self):
        """
        PyVista UnstructuredGrid を構築して返す。

        Returns
        -------
        pyvista.UnstructuredGrid
        """
        import pyvista as pv

        pos = self.get_positions()

        if self.element_type not in _VTK_CELL_TYPES:
            raise ValueError(f"未対応の要素タイプ: {self.element_type}")

        vtk_type = _VTK_CELL_TYPES[self.element_type]

        cells = []
        cell_types = []
        for node_ids in self.elements.values():
            idx = [self._node_id_to_idx[nid] for nid in node_ids]
            cells.append(len(idx))
            cells.extend(idx)
            cell_types.append(vtk_type)

        cells_arr = np.array(cells, dtype=np.int64)
        types_arr = np.array(cell_types, dtype=np.uint8)

        return pv.UnstructuredGrid(cells_arr, types_arr, pos)

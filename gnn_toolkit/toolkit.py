"""
GNNToolkit — 学習・推論・保存・評価を統合するファサードクラス

v3.1: 荷重方向ベクトル化で引張・曲げ等の複数荷重方向に対応。

使い方:
    tk = GNNToolkit(train_load=1000.0)
    tk.train("result_1000N.vtu")  # 荷重方向は変位から自動検出
    tk.predict("result_1000N.vtu", load_N=500.0)
    tk.save("saved_model")
    tk.load("saved_model")

    # 複数荷重方向学習（引張 + 曲げ）
    tk.train(
        ["tension_1000N.vtu", "bending_1000N.vtu"],
        load_values=[1000, 1000],
    )  # 荷重方向は各VTUの変位から自動検出

    # CalculiX .inp ファイルから推論（荷重方向はCLOADから自動取得）
    tk.predict("FEMMeshNetgen.inp")
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F

from .config import GNNConfig
from .data import FEADataProcessor
from .model import StructuralGNN


class GNNToolkit:
    """汎用構造解析 GNN のファサード。"""

    def __init__(self, data_dir: str = "data", results_dir: str = "results", **kwargs) -> None:
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        self.config = GNNConfig(**kwargs)
        self.config.update_n_features()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[StructuralGNN] = None
        self._loss_history: List[float] = []
        mode = "ジオメトリ" if self.config.include_geometry else "レガシー"
        print(f"[GNNToolkit] device={self.device}, mode={mode}")
        print(f"  data_dir    = {os.path.abspath(self.data_dir)}")
        print(f"  results_dir = {os.path.abspath(self.results_dir)}")

    # ==================================================================
    # 学習
    # ==================================================================
    def train(
        self,
        vtu_files: Union[str, List[str]],
        load_values: Optional[Union[float, List[float]]] = None,
        *,
        load_directions: Optional[List[Optional[np.ndarray]]] = None,
        callback=None,
    ) -> List[float]:
        """
        VTU ファイルから GNN モデルを学習する。

        Parameters
        ----------
        vtu_files : str | list[str]
            学習用 VTU ファイル（パス）。複数指定で形状汎化学習。
        load_values : float | list[float], optional
            各ファイルの荷重値 [N]（省略時は config.train_load）
        load_directions : list[ndarray(3,)], optional
            各ファイルの荷重方向単位ベクトル。None の場合は変位から自動検出。
        callback : callable, optional
            ``callback(epoch, loss, best_loss, lr)`` — epoch ごとに呼ばれる

        Returns
        -------
        list[float]
            epoch ごとの合計 Loss 履歴
        """
        if isinstance(vtu_files, str):
            vtu_files = [vtu_files]
        vtu_files = [self._resolve_data(f) for f in vtu_files]
        if load_values is None:
            load_values = [self.config.train_load] * len(vtu_files)
        elif isinstance(load_values, (int, float)):
            load_values = [float(load_values)] * len(vtu_files)
        if len(load_values) == 1 and len(vtu_files) > 1:
            load_values = load_values * len(vtu_files)
        if load_directions is None:
            load_directions = [None] * len(vtu_files)

        # Step 1 — キャリブレーション
        print("=" * 60)
        print("[Step 1] データ解析 & 自動キャリブレーション")
        print("=" * 60)
        meshes = [FEADataProcessor.analyze(f) for f in vtu_files]
        if len(meshes) == 1:
            self.config.auto_calibrate(meshes[0], load_values[0])
        else:
            self.config.auto_calibrate_multi(meshes, load_values)
        self.config.update_n_features()
        print(self.config.summary())

        # Step 2 — モデル構築
        self.model = StructuralGNN(self.config).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"\n[Step 2] モデル構築: {self.config.n_layers}層 SAGEConv "
            f"(hidden={self.config.hidden_dim}, features={self.config.n_features}, "
            f"params={n_params:,})"
        )

        # Step 3 — データ準備
        datasets = [
            FEADataProcessor.to_graph(f, lv, self.config, ld).to(self.device)
            for f, lv, ld in zip(vtu_files, load_values, load_directions)
        ]
        print(f"  学習ファイル数: {len(datasets)}")
        for f, d in zip(vtu_files, datasets):
            print(f"    {os.path.basename(f)}: nodes={d.x.shape[0]}, "
                  f"edges={d.edge_index.shape[1]//2}")

        # Step 4 — 学習ループ
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs, eta_min=self.config.lr_min
        )

        self.model.train()
        best_loss, wait = float("inf"), 0
        best_state = None
        self._loss_history = []

        # 荷重ベクトル列インデックス (Fx, Fy, Fz)
        if self.config.include_geometry:
            load_cols = [11, 12, 13]  # geo(9) + is_fixed(1) + is_load(1) → 11,12,13
        else:
            load_cols = [5, 6, 7]     # pos(3) + is_fixed(1) + is_load(1) → 5,6,7

        print(
            f"\n[Step 3] 学習開始 "
            f"(epochs={self.config.epochs}, patience={self.config.patience})"
        )
        print("-" * 60)

        for epoch in range(self.config.epochs + 1):
            optimizer.zero_grad()
            total_loss = 0.0

            for base_data in datasets:
                for r in self.config.load_ratios:
                    cd = base_data.clone()
                    # 荷重ベクトル (Fx, Fy, Fz) をスケーリング
                    for col in load_cols:
                        mask = cd.x[:, col].abs() > 0
                        cd.x[mask, col] = base_data.x[mask, col] * r
                    cd.y = base_data.y * r

                    out = self.model(cd)
                    loss_d = F.mse_loss(out[:, :3], cd.y[:, :3])
                    loss_s = F.mse_loss(out[:, 3], cd.y[:, 3])
                    loss = loss_d + loss_s * self.config.stress_weight
                    loss.backward()
                    total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            self._loss_history.append(total_loss)

            if total_loss < best_loss:
                best_loss = total_loss
                wait = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1

            lr_now = optimizer.param_groups[0]["lr"]

            if callback:
                callback(epoch, total_loss, best_loss, lr_now)

            if epoch % 500 == 0:
                print(
                    f"  Epoch {epoch:5d} | Loss {total_loss:.6f} "
                    f"| Best {best_loss:.6f} | LR {lr_now:.2e}"
                )

            if wait >= self.config.patience and epoch > 1000:
                print(
                    f"  >>> Early Stop @ epoch {epoch} "
                    f"(patience={self.config.patience})"
                )
                break

        # ベストモデルを復元
        if best_state is not None:
            self.model.load_state_dict(best_state)

        print("-" * 60)
        print(f"[学習完了] Best Loss = {best_loss:.6f}")
        return self._loss_history

    # ==================================================================
    # 推論
    # ==================================================================
    def predict(
        self,
        mesh_file: str,
        load_N: Optional[float] = None,
        output_vtu: Optional[str] = None,
        load_direction: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        学習済みモデルで推論し VTU に保存する。

        VTU または CalculiX .inp ファイルを入力として受け取る。
        .inp ファイルの場合は拘束・荷重条件をファイルから直接読み取る。

        Parameters
        ----------
        mesh_file : str
            VTU ファイルまたは CalculiX .inp ファイルのパス
        load_N : float, optional
            荷重値 [N]。VTU の場合は省略時 train_load,
            .inp の場合は省略時 CLOAD から自動計算
        output_vtu : str, optional
            出力 VTU ファイル名（省略時は自動生成）
        load_direction : ndarray (3,), optional
            荷重方向の単位ベクトル。None の場合は自動推定。

        Returns
        -------
        dict  — max_disp, max_disp_x/y/z, max_stress, output
        """
        assert self.model is not None, "先に train() または load() を実行してください"

        # .inp ファイルの場合は専用処理
        if mesh_file.lower().endswith(".inp"):
            return self._predict_from_inp(mesh_file, load_N, output_vtu, load_direction)

        # --- VTU ファイル処理 ---
        vtu_file = self._resolve_data(mesh_file)
        if load_N is None:
            load_N = self.config.train_load
        if output_vtu is None:
            output_vtu = f"gnn_{int(load_N)}N_result.vtu"
        output_vtu = self._resolve_results(output_vtu)

        data = FEADataProcessor.to_graph(
            vtu_file, load_N, self.config
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(data).cpu().numpy()

        mesh = pv.read(vtu_file)
        mesh = mesh.cell_data_to_point_data()
        disp_mm = pred[:, :3] * self.config.norm_disp
        stress_mpa = pred[:, 3] * self.config.norm_stress

        mesh.point_data["gnn_disp"] = disp_mm
        mesh.point_data["gnn_disp_x"] = disp_mm[:, 0]
        mesh.point_data["gnn_disp_y"] = disp_mm[:, 1]
        mesh.point_data["gnn_disp_z"] = disp_mm[:, 2]
        mesh.point_data["gnn_stress"] = stress_mpa
        mesh.save(output_vtu)

        max_dx = float(np.max(np.abs(disp_mm[:, 0])))
        max_dy = float(np.max(np.abs(disp_mm[:, 1])))
        max_dz = float(np.max(np.abs(disp_mm[:, 2])))
        res = {
            "max_disp": float(np.max(np.abs(disp_mm))),
            "max_disp_x": max_dx,
            "max_disp_y": max_dy,
            "max_disp_z": max_dz,
            "max_stress": float(np.max(stress_mpa)),
            "output": output_vtu,
        }
        print(f"\n--- [{load_N}N 推論結果] ---")
        print(f"  最大変位   : {res['max_disp']:.5f} mm")
        print(f"    X        : {max_dx:.5f} mm")
        print(f"    Y        : {max_dy:.5f} mm")
        print(f"    Z        : {max_dz:.5f} mm")
        print(f"  最大応力   : {res['max_stress']:.5f} MPa")
        print(f"  保存先     : {output_vtu}")
        return res

    # ==================================================================
    # .inp 推論
    # ==================================================================
    def _predict_from_inp(
        self,
        inp_file: str,
        load_N: Optional[float] = None,
        output_vtu: Optional[str] = None,
        load_direction: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """CalculiX .inp ファイルから推論する。"""
        inp_file = self._resolve_data(inp_file)

        data, reader = FEADataProcessor.to_graph_from_inp(
            inp_file, load_N, self.config, load_direction
        )

        # load_N が None の場合はファイルから取得済み
        if load_N is None:
            load_N = reader.get_total_load()

        if output_vtu is None:
            output_vtu = f"gnn_{int(load_N)}N_result.vtu"
        output_vtu = self._resolve_results(output_vtu)

        print(f"\n--- [.inp ファイル解析] ---")
        print(reader.summary())

        data = data.to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(data).cpu().numpy()

        # PyVista メッシュ構築 & 結果保存
        mesh = reader.to_pyvista()
        disp_mm = pred[:, :3] * self.config.norm_disp
        stress_mpa = pred[:, 3] * self.config.norm_stress

        mesh.point_data["gnn_disp"] = disp_mm
        mesh.point_data["gnn_disp_x"] = disp_mm[:, 0]
        mesh.point_data["gnn_disp_y"] = disp_mm[:, 1]
        mesh.point_data["gnn_disp_z"] = disp_mm[:, 2]
        mesh.point_data["gnn_stress"] = stress_mpa
        mesh.save(output_vtu)

        max_dx = float(np.max(np.abs(disp_mm[:, 0])))
        max_dy = float(np.max(np.abs(disp_mm[:, 1])))
        max_dz = float(np.max(np.abs(disp_mm[:, 2])))
        res = {
            "max_disp": float(np.max(np.abs(disp_mm))),
            "max_disp_x": max_dx,
            "max_disp_y": max_dy,
            "max_disp_z": max_dz,
            "max_stress": float(np.max(stress_mpa)),
            "output": output_vtu,
        }
        print(f"\n--- [{load_N}N 推論結果 (from .inp)] ---")
        print(f"  最大変位   : {res['max_disp']:.5f} mm")
        print(f"    X        : {max_dx:.5f} mm")
        print(f"    Y        : {max_dy:.5f} mm")
        print(f"    Z        : {max_dz:.5f} mm")
        print(f"  最大応力   : {res['max_stress']:.5f} MPa")
        print(f"  保存先     : {output_vtu}")
        return res

    # ==================================================================
    # 精度評価
    # ==================================================================
    def evaluate(
        self,
        vtu_file: str,
        load_N: Optional[float] = None,
    ) -> Dict[str, float]:
        """学習データ（正解）との誤差を算出する。"""
        assert self.model is not None, "先に train() または load() を実行してください"
        vtu_file = self._resolve_data(vtu_file)
        if load_N is None:
            load_N = self.config.train_load

        data = FEADataProcessor.to_graph(
            vtu_file, load_N, self.config
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(data).cpu().numpy()
        true = data.y.cpu().numpy()

        dp = pred[:, :3] * self.config.norm_disp
        dt = true[:, :3] * self.config.norm_disp
        sp = pred[:, 3] * self.config.norm_stress
        st = true[:, 3] * self.config.norm_stress

        # 全体指標
        d_mae = float(np.mean(np.abs(dp - dt)))
        d_max = float(np.max(np.abs(dp - dt)))
        s_mae = float(np.mean(np.abs(sp - st)))
        s_max = float(np.max(np.abs(sp - st)))
        d_rel = d_max / max(np.max(np.abs(dt)), 1e-12) * 100
        s_rel = s_max / max(np.max(np.abs(st)), 1e-12) * 100

        # 各方向指標
        axis_labels = ["X", "Y", "Z"]
        axis_results = {}
        for i, ax in enumerate(axis_labels):
            ax_mae = float(np.mean(np.abs(dp[:, i] - dt[:, i])))
            ax_max = float(np.max(np.abs(dp[:, i] - dt[:, i])))
            ax_ref = max(float(np.max(np.abs(dt[:, i]))), 1e-12)
            ax_rel = ax_max / ax_ref * 100
            axis_results[f"d_mae_{ax.lower()}"] = ax_mae
            axis_results[f"d_max_{ax.lower()}"] = ax_max
            axis_results[f"d_rel_{ax.lower()}"] = ax_rel

        print(f"\n--- [精度評価 {load_N}N] ---")
        for ax in axis_labels:
            k = ax.lower()
            print(
                f"  変位{ax} MAE = {axis_results[f'd_mae_{k}']:.6f} mm  |  "
                f"最大誤差 = {axis_results[f'd_max_{k}']:.6f} mm  "
                f"({axis_results[f'd_rel_{k}']:.2f}%)"
            )
        print(
            f"  変位  MAE = {d_mae:.6f} mm  |  "
            f"最大誤差 = {d_max:.6f} mm  ({d_rel:.2f}%)"
        )
        print(
            f"  応力  MAE = {s_mae:.4f} MPa  |  "
            f"最大誤差 = {s_max:.4f} MPa  ({s_rel:.2f}%)"
        )
        return {
            "d_mae": d_mae, "d_max": d_max, "d_rel": float(d_rel),
            "s_mae": s_mae, "s_max": s_max, "s_rel": float(s_rel),
            **axis_results,
        }

    # ==================================================================
    # 保存・読込
    # ==================================================================
    def save(self, directory: str) -> None:
        """モデルと設定を保存する。"""
        os.makedirs(directory, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(directory, "model.pth"),
        )
        self.config.save(os.path.join(directory, "config.json"))
        print(f"[保存完了] {directory}/")

    def load(self, directory: str) -> None:
        """保存済みモデルを読み込む。v2 / v3 両モデル対応。"""
        self.config = GNNConfig.load(os.path.join(directory, "config.json"))
        self.config.update_n_features()
        self.model = StructuralGNN(self.config).to(self.device)
        self.model.load_state_dict(
            torch.load(
                os.path.join(directory, "model.pth"),
                map_location=self.device,
                weights_only=True,
            )
        )
        mode = "ジオメトリ" if self.config.include_geometry else "レガシー"
        print(f"[読込完了] {directory}/ (mode={mode}, "
              f"features={self.config.n_features})")

    # ==================================================================
    # パス解決ヘルパー
    # ==================================================================
    def _resolve_data(self, path: str) -> str:
        """ファイルが存在しなければ data_dir 配下を探す。"""
        if os.path.isfile(path):
            return path
        joined = os.path.join(self.data_dir, os.path.basename(path))
        if os.path.isfile(joined):
            return joined
        return path

    def _resolve_results(self, path: str) -> str:
        """出力パスを results_dir 配下に配置する。"""
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return path
        return os.path.join(self.results_dir, path)

    @property
    def loss_history(self) -> List[float]:
        """直近の学習 Loss 履歴を返す。"""
        return self._loss_history

    @property
    def is_trained(self) -> bool:
        return self.model is not None

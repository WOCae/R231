"""
GNNToolkitUI — ipywidgets ベースの対話型 GUI

Jupyter Notebook 上でボタン・スライダー・ドロップダウンを使い、
学習 → 推論 → 評価 → 保存/読込を直観的に操作できる。
"""

from __future__ import annotations

import glob
import os
import threading
from typing import Optional

import ipywidgets as widgets
import matplotlib
matplotlib.use("agg")          # Colab バックエンド互換
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

from .toolkit import GNNToolkit


# ======================================================================
# ヘルパー
# ======================================================================
def _vtu_files(directory: str = ".") -> list:
    """カレントディレクトリの .vtu ファイルを返す。"""
    return sorted(glob.glob(os.path.join(directory, "*.vtu")))


def _model_dirs(directory: str = ".") -> list:
    """config.json が存在するサブディレクトリを返す。"""
    dirs = []
    for d in sorted(os.listdir(directory)):
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json")):
            dirs.append(d)
    return dirs


# ======================================================================
# メイン UI クラス
# ======================================================================
class GNNToolkitUI:
    """
    Jupyter 上で GNNToolkit を操作する対話型 GUI。

    使い方::

        from gnn_toolkit import GNNToolkitUI
        ui = GNNToolkitUI()
        ui.show()
    """

    # スタイル定数
    _BUTTON_LAYOUT = widgets.Layout(width="160px", height="36px")
    _WIDE_LAYOUT = widgets.Layout(width="320px")
    _LOG_LAYOUT = widgets.Layout(
        width="100%", height="280px",
        border="1px solid #ccc",
        overflow_y="auto",
    )

    def __init__(self, work_dir: str = ".") -> None:
        self.work_dir = work_dir
        self.tk: Optional[GNNToolkit] = None
        self._build_widgets()

    # ==================================================================
    # ウィジェット構築
    # ==================================================================
    def _build_widgets(self) -> None:
        # ─── ヘッダー ─────────────────────────────────
        header = widgets.HTML(
            "<h2 style='margin:0 0 4px 0;'>🔧 構造解析 GNN ツールキット</h2>"
            "<p style='color:#666; margin:0 0 12px 0;'>"
            "VTU から学習 → 任意荷重で推論 → モデル保存/読込</p>"
        )

        # ─── タブ 1: 学習 ─────────────────────────────
        vtu_list = _vtu_files(self.work_dir)
        self.w_train_file = widgets.Dropdown(
            options=vtu_list,
            description="学習VTU:",
            layout=self._WIDE_LAYOUT,
        )
        self.w_train_load = widgets.FloatText(
            value=1000.0, description="基準荷重[N]:",
            layout=self._WIDE_LAYOUT,
        )
        self.w_epochs = widgets.IntSlider(
            value=5000, min=500, max=20000, step=500,
            description="Epochs:", layout=self._WIDE_LAYOUT,
            style={"description_width": "80px"},
        )
        self.w_hidden = widgets.Dropdown(
            options=[64, 128, 256, 512],
            value=128, description="Hidden:",
            layout=self._WIDE_LAYOUT,
        )
        self.w_layers = widgets.IntSlider(
            value=4, min=2, max=8, step=1,
            description="GNN層数:", layout=self._WIDE_LAYOUT,
            style={"description_width": "80px"},
        )
        self.w_stress_wt = widgets.FloatLogSlider(
            value=100.0, base=10, min=0, max=4, step=0.5,
            description="応力重み:", layout=self._WIDE_LAYOUT,
            style={"description_width": "80px"},
        )
        self.w_patience = widgets.IntSlider(
            value=500, min=100, max=2000, step=100,
            description="Patience:", layout=self._WIDE_LAYOUT,
            style={"description_width": "80px"},
        )
        self.w_lr = widgets.FloatLogSlider(
            value=0.001, base=10, min=-5, max=-2, step=0.5,
            description="学習率:", layout=self._WIDE_LAYOUT,
            style={"description_width": "80px"},
        )
        self.w_linear = widgets.Checkbox(
            value=True, description="線形弾性スケーリング",
        )

        self.btn_train = widgets.Button(
            description="▶ 学習開始",
            button_style="primary",
            layout=self._BUTTON_LAYOUT,
            icon="play",
        )
        self.btn_train.on_click(self._on_train)

        self.w_progress = widgets.IntProgress(
            value=0, min=0, max=100,
            description="進捗:",
            bar_style="info",
            layout=widgets.Layout(width="100%"),
        )
        self.w_progress.layout.visibility = "hidden"

        train_params_left = widgets.VBox([
            self.w_train_file,
            self.w_train_load,
            self.w_epochs,
            self.w_hidden,
        ])
        train_params_right = widgets.VBox([
            self.w_layers,
            self.w_stress_wt,
            self.w_patience,
            self.w_lr,
            self.w_linear,
        ])
        train_tab = widgets.VBox([
            widgets.HBox([train_params_left, train_params_right]),
            widgets.HBox([self.btn_train]),
            self.w_progress,
        ])

        # ─── タブ 2: 推論 ─────────────────────────────
        self.w_pred_file = widgets.Dropdown(
            options=vtu_list,
            description="元VTU:",
            layout=self._WIDE_LAYOUT,
        )
        self.w_pred_load = widgets.FloatText(
            value=500.0, description="荷重[N]:",
            layout=self._WIDE_LAYOUT,
        )
        self.w_pred_output = widgets.Text(
            value="", description="出力名:",
            placeholder="(自動: gnn_500N_result.vtu)",
            layout=self._WIDE_LAYOUT,
        )

        self.btn_predict = widgets.Button(
            description="▶ 推論実行",
            button_style="success",
            layout=self._BUTTON_LAYOUT,
            icon="bolt",
        )
        self.btn_predict.on_click(self._on_predict)

        predict_tab = widgets.VBox([
            self.w_pred_file,
            self.w_pred_load,
            self.w_pred_output,
            self.btn_predict,
        ])

        # ─── タブ 3: 評価 ─────────────────────────────
        self.w_eval_file = widgets.Dropdown(
            options=vtu_list,
            description="VTU:",
            layout=self._WIDE_LAYOUT,
        )
        self.w_eval_load = widgets.FloatText(
            value=1000.0, description="荷重[N]:",
            layout=self._WIDE_LAYOUT,
        )
        self.btn_evaluate = widgets.Button(
            description="📊 精度評価",
            button_style="info",
            layout=self._BUTTON_LAYOUT,
            icon="chart-bar",
        )
        self.btn_evaluate.on_click(self._on_evaluate)

        self.btn_plot_loss = widgets.Button(
            description="📈 Loss曲線",
            button_style="",
            layout=self._BUTTON_LAYOUT,
            icon="line-chart",
        )
        self.btn_plot_loss.on_click(self._on_plot_loss)

        eval_tab = widgets.VBox([
            self.w_eval_file,
            self.w_eval_load,
            widgets.HBox([self.btn_evaluate, self.btn_plot_loss]),
        ])

        # ─── タブ 4: 保存/読込 ────────────────────────
        self.w_save_dir = widgets.Text(
            value="saved_model", description="保存先:",
            layout=self._WIDE_LAYOUT,
        )
        self.btn_save = widgets.Button(
            description="💾 保存",
            button_style="warning",
            layout=self._BUTTON_LAYOUT,
            icon="save",
        )
        self.btn_save.on_click(self._on_save)

        model_dirs = _model_dirs(self.work_dir)
        self.w_load_dir = widgets.Dropdown(
            options=model_dirs if model_dirs else ["(なし)"],
            description="読込元:",
            layout=self._WIDE_LAYOUT,
        )
        self.btn_load = widgets.Button(
            description="📂 読込",
            button_style="",
            layout=self._BUTTON_LAYOUT,
            icon="folder-open",
        )
        self.btn_load.on_click(self._on_load)
        self.btn_refresh = widgets.Button(
            description="🔄 更新",
            button_style="",
            layout=widgets.Layout(width="80px", height="36px"),
            icon="refresh",
        )
        self.btn_refresh.on_click(self._on_refresh)

        save_tab = widgets.VBox([
            widgets.HTML("<b>モデル保存</b>"),
            widgets.HBox([self.w_save_dir, self.btn_save]),
            widgets.HTML("<hr><b>モデル読込</b>"),
            widgets.HBox([self.w_load_dir, self.btn_load, self.btn_refresh]),
        ])

        # ─── タブ 5: VTU解析 ──────────────────────────
        self.w_analyze_file = widgets.Dropdown(
            options=vtu_list,
            description="VTU:",
            layout=self._WIDE_LAYOUT,
        )
        self.btn_analyze = widgets.Button(
            description="🔍 解析",
            button_style="",
            layout=self._BUTTON_LAYOUT,
            icon="search",
        )
        self.btn_analyze.on_click(self._on_analyze)

        analyze_tab = widgets.VBox([
            self.w_analyze_file,
            self.btn_analyze,
        ])

        # ─── アコーディオン組み立て（Colab 互換）────────
        self.tabs = widgets.Accordion(
            children=[train_tab, predict_tab, eval_tab, save_tab, analyze_tab]
        )
        for i, label in enumerate(["▶ 学習", "⚡ 推論", "📊 評価", "💾 保存/読込", "🔍 VTU解析"]):
            self.tabs.set_title(i, label)
        self.tabs.selected_index = 0

        # ─── ステータスバー ───────────────────────────
        self.w_status = widgets.HTML(
            value="<i style='color:#888;'>待機中</i>"
        )

        # ─── ログ出力 ─────────────────────────────────
        self.out = widgets.Output(layout=self._LOG_LAYOUT)

        # ─── 全体レイアウト ───────────────────────────
        self.ui = widgets.VBox([
            header,
            self.tabs,
            self.w_status,
            self.out,
        ])

    # ==================================================================
    # 表示
    # ==================================================================
    def show(self) -> None:
        """UI を表示する。"""
        display(self.ui)

    # ==================================================================
    # コールバック
    # ==================================================================
    def _set_status(self, msg: str, color: str = "#333") -> None:
        self.w_status.value = f"<b style='color:{color};'>{msg}</b>"

    def _ensure_toolkit(self) -> None:
        """GNNToolkit が未初期化なら生成する。"""
        if self.tk is None:
            with self.out:
                self.tk = GNNToolkit(train_load=self.w_train_load.value)

    # ---- 学習 --------------------------------------------------------
    def _on_train(self, _) -> None:
        self.out.clear_output()
        with self.out:
            vtu = self.w_train_file.value
            if not vtu:
                self._set_status("VTU ファイルを選択してください", "red")
                return
            self._set_status("学習中...", "blue")
            self.w_progress.layout.visibility = "visible"
            self.w_progress.max = self.w_epochs.value
            self.w_progress.value = 0

            self.tk = GNNToolkit(
                train_load=self.w_train_load.value,
                epochs=self.w_epochs.value,
                hidden_dim=self.w_hidden.value,
                n_layers=self.w_layers.value,
                stress_weight=self.w_stress_wt.value,
                patience=self.w_patience.value,
                lr=self.w_lr.value,
                linear_scaling=self.w_linear.value,
            )

            def _cb(epoch, loss, best, lr):
                self.w_progress.value = min(epoch, self.w_progress.max)

            self.tk.train(vtu, callback=_cb)
            self.w_progress.value = self.w_progress.max
            self._set_status("学習完了 ✓", "green")

    # ---- 推論 --------------------------------------------------------
    def _on_predict(self, _) -> None:
        self.out.clear_output()
        with self.out:
            if self.tk is None or not self.tk.is_trained:
                self._set_status("先にモデルを学習または読込してください", "red")
                return
            vtu = self.w_pred_file.value
            load_n = self.w_pred_load.value
            out_name = self.w_pred_output.value.strip() or None
            self._set_status("推論中...", "blue")
            res = self.tk.predict(vtu, load_n, out_name)
            self._set_status(
                f"推論完了 — 最大変位 {res['max_disp']:.5f} mm / "
                f"最大応力 {res['max_stress']:.3f} MPa",
                "green",
            )
            # ファイル一覧を更新
            self._refresh_vtu_lists()

    # ---- 評価 --------------------------------------------------------
    def _on_evaluate(self, _) -> None:
        self.out.clear_output()
        with self.out:
            if self.tk is None or not self.tk.is_trained:
                self._set_status("先にモデルを学習または読込してください", "red")
                return
            self._set_status("評価中...", "blue")
            res = self.tk.evaluate(
                self.w_eval_file.value,
                self.w_eval_load.value,
            )
            self._set_status(
                f"評価完了 — 変位誤差 {res['d_rel']:.2f}% / "
                f"応力誤差 {res['s_rel']:.2f}%",
                "green",
            )

    # ---- Loss 曲線 ---------------------------------------------------
    def _on_plot_loss(self, _) -> None:
        self.out.clear_output()
        with self.out:
            if self.tk is None or not self.tk.loss_history:
                self._set_status("学習履歴がありません", "red")
                return
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.semilogy(self.tk.loss_history)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Total Loss")
            ax.set_title("学習 Loss 曲線")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            self._set_status("Loss 曲線を表示しました", "green")

    # ---- 保存 --------------------------------------------------------
    def _on_save(self, _) -> None:
        self.out.clear_output()
        with self.out:
            if self.tk is None or not self.tk.is_trained:
                self._set_status("保存するモデルがありません", "red")
                return
            d = self.w_save_dir.value.strip()
            if not d:
                self._set_status("保存先を入力してください", "red")
                return
            self.tk.save(d)
            self._set_status(f"モデルを {d}/ に保存しました", "green")
            self._refresh_model_dirs()

    # ---- 読込 --------------------------------------------------------
    def _on_load(self, _) -> None:
        self.out.clear_output()
        with self.out:
            d = self.w_load_dir.value
            if not d or d == "(なし)":
                self._set_status("読込元を選択してください", "red")
                return
            self._ensure_toolkit()
            self.tk.load(d)
            self._set_status(f"モデルを {d}/ から読み込みました", "green")

    # ---- 更新 --------------------------------------------------------
    def _on_refresh(self, _) -> None:
        self._refresh_vtu_lists()
        self._refresh_model_dirs()
        self._set_status("ファイル一覧を更新しました", "#333")

    # ---- VTU 解析 ----------------------------------------------------
    def _on_analyze(self, _) -> None:
        self.out.clear_output()
        with self.out:
            vtu = self.w_analyze_file.value
            if not vtu:
                self._set_status("VTU ファイルを選択してください", "red")
                return
            from .data import FEADataProcessor
            FEADataProcessor.analyze(vtu)
            self._set_status(f"{vtu} の解析完了", "green")

    # ---- リフレッシュ ------------------------------------------------
    def _refresh_vtu_lists(self) -> None:
        vtu_list = _vtu_files(self.work_dir)
        for w in [
            self.w_train_file,
            self.w_pred_file,
            self.w_eval_file,
            self.w_analyze_file,
        ]:
            w.options = vtu_list

    def _refresh_model_dirs(self) -> None:
        dirs = _model_dirs(self.work_dir)
        self.w_load_dir.options = dirs if dirs else ["(なし)"]

"""
GNNToolkitUI — ipywidgets ベースの対話型 GUI

Colab (ipywidgets 7.x) / ローカル (8.x) 両対応。
Tab / Accordion は Colab で描画されないため、
ToggleButtons + Output でページ切替を行う。
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import ipywidgets as widgets
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from .toolkit import GNNToolkit


# ======================================================================
# ヘルパー
# ======================================================================
def _vtu_files(directory: str = ".") -> list:
    return sorted(glob.glob(os.path.join(directory, "*.vtu")))


def _model_dirs(directory: str = ".") -> list:
    dirs = []
    for d in sorted(os.listdir(directory)):
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json")):
            dirs.append(d)
    return dirs


# ======================================================================
# メイン UI
# ======================================================================
class GNNToolkitUI:
    """ToggleButtons ベースの Colab 互換 GUI。"""

    _BTN = widgets.Layout(width="160px", height="36px")
    _WIDE = widgets.Layout(width="320px")
    _LOG = widgets.Layout(
        width="100%", height="280px",
        border="1px solid #ccc", overflow_y="auto",
    )

    def __init__(self, work_dir: str = ".",
                 data_dir: str = "data",
                 results_dir: str = "results") -> None:
        self.work_dir = work_dir
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.tk: Optional[GNNToolkit] = None
        self._build()

    # ==================================================================
    # 構築
    # ==================================================================
    def _build(self) -> None:
        header = widgets.HTML(
            "<h2 style='margin:0 0 4px 0;'>🔧 構造解析 GNN ツールキット</h2>"
            "<p style='color:#666;margin:0 0 8px 0;'>"
            "VTU から学習 → 任意荷重で推論 → モデル保存/読込</p>"
        )

        vtu_list = _vtu_files(self.data_dir)

        # ── 各ページのウィジェット ────────────────────

        # 1) 学習
        self.w_train_file = widgets.Dropdown(options=vtu_list, description="学習VTU:", layout=self._WIDE)
        self.w_train_load = widgets.FloatText(value=1000.0, description="基準荷重[N]:", layout=self._WIDE)
        self.w_epochs = widgets.IntSlider(value=5000, min=500, max=20000, step=500,
                                          description="Epochs:", layout=self._WIDE,
                                          style={"description_width": "80px"})
        self.w_hidden = widgets.Dropdown(options=[64, 128, 256, 512], value=128,
                                         description="Hidden:", layout=self._WIDE)
        self.w_layers = widgets.IntSlider(value=4, min=2, max=8, description="GNN層数:",
                                          layout=self._WIDE, style={"description_width": "80px"})
        self.w_stress_wt = widgets.FloatLogSlider(value=100.0, base=10, min=0, max=4, step=0.5,
                                                   description="応力重み:", layout=self._WIDE,
                                                   style={"description_width": "80px"})
        self.w_patience = widgets.IntSlider(value=500, min=100, max=2000, step=100,
                                            description="Patience:", layout=self._WIDE,
                                            style={"description_width": "80px"})
        self.w_lr = widgets.FloatLogSlider(value=0.001, base=10, min=-5, max=-2, step=0.5,
                                           description="学習率:", layout=self._WIDE,
                                           style={"description_width": "80px"})
        self.w_linear = widgets.Checkbox(value=True, description="線形弾性スケーリング")
        self.btn_train = widgets.Button(description="▶ 学習開始", button_style="primary", layout=self._BTN)
        self.btn_train.on_click(self._on_train)
        self.w_progress = widgets.IntProgress(value=0, min=0, max=100, description="進捗:",
                                               bar_style="info",
                                               layout=widgets.Layout(width="100%"))
        self.w_progress.layout.visibility = "hidden"

        page_train = widgets.VBox([
            widgets.HBox([
                widgets.VBox([self.w_train_file, self.w_train_load, self.w_epochs, self.w_hidden]),
                widgets.VBox([self.w_layers, self.w_stress_wt, self.w_patience, self.w_lr, self.w_linear]),
            ]),
            self.btn_train,
            self.w_progress,
        ])

        # 2) 推論
        self.w_pred_file = widgets.Dropdown(options=vtu_list, description="元VTU:", layout=self._WIDE)
        self.w_pred_load = widgets.FloatText(value=500.0, description="荷重[N]:", layout=self._WIDE)
        self.w_pred_output = widgets.Text(value="", description="出力名:",
                                          placeholder="(自動)", layout=self._WIDE)
        self.btn_predict = widgets.Button(description="⚡ 推論実行", button_style="success", layout=self._BTN)
        self.btn_predict.on_click(self._on_predict)
        page_predict = widgets.VBox([self.w_pred_file, self.w_pred_load, self.w_pred_output, self.btn_predict])

        # 3) 評価
        self.w_eval_file = widgets.Dropdown(options=vtu_list, description="VTU:", layout=self._WIDE)
        self.w_eval_load = widgets.FloatText(value=1000.0, description="荷重[N]:", layout=self._WIDE)
        self.btn_evaluate = widgets.Button(description="📊 精度評価", button_style="info", layout=self._BTN)
        self.btn_evaluate.on_click(self._on_evaluate)
        self.btn_plot_loss = widgets.Button(description="📈 Loss曲線", layout=self._BTN)
        self.btn_plot_loss.on_click(self._on_plot_loss)
        page_eval = widgets.VBox([self.w_eval_file, self.w_eval_load,
                                  widgets.HBox([self.btn_evaluate, self.btn_plot_loss])])

        # 4) 保存/読込
        self.w_save_dir = widgets.Text(value="saved_model", description="保存先:", layout=self._WIDE)
        self.btn_save = widgets.Button(description="💾 保存", button_style="warning", layout=self._BTN)
        self.btn_save.on_click(self._on_save)
        model_dirs = _model_dirs(self.work_dir)
        self.w_load_dir = widgets.Dropdown(
            options=model_dirs if model_dirs else ["(なし)"],
            description="読込元:", layout=self._WIDE)
        self.btn_load = widgets.Button(description="📂 読込", layout=self._BTN)
        self.btn_load.on_click(self._on_load)
        self.btn_refresh = widgets.Button(description="🔄 更新",
                                          layout=widgets.Layout(width="80px", height="36px"))
        self.btn_refresh.on_click(self._on_refresh)
        page_save = widgets.VBox([
            widgets.HTML("<b>モデル保存</b>"),
            widgets.HBox([self.w_save_dir, self.btn_save]),
            widgets.HTML("<hr><b>モデル読込</b>"),
            widgets.HBox([self.w_load_dir, self.btn_load, self.btn_refresh]),
        ])

        # 5) VTU解析
        self.w_analyze_file = widgets.Dropdown(options=vtu_list, description="VTU:", layout=self._WIDE)
        self.btn_analyze = widgets.Button(description="🔍 VTU解析", layout=self._BTN)
        self.btn_analyze.on_click(self._on_analyze)
        page_analyze = widgets.VBox([self.w_analyze_file, self.btn_analyze])

        # ── ページ切替（ToggleButtons + Output）────────
        self._pages = {
            "学習": page_train,
            "推論": page_predict,
            "評価": page_eval,
            "保存/読込": page_save,
            "VTU解析": page_analyze,
        }
        self.w_nav = widgets.ToggleButtons(
            options=list(self._pages.keys()),
            description="",
            button_style="info",
            style={"button_width": "100px"},
        )
        self.w_nav.observe(self._on_nav, names="value")

        self._page_area = widgets.Output()
        self._show_page(self.w_nav.value)

        # ── ステータス & ログ ──────────────────────────
        self.w_status = widgets.HTML("<i style='color:#888;'>待機中</i>")
        self.out = widgets.Output(layout=self._LOG)

        # ── 全体レイアウト ─────────────────────────────
        self.ui = widgets.VBox([
            header,
            self.w_nav,
            self._page_area,
            self.w_status,
            self.out,
        ])

    # ==================================================================
    # ページ切替
    # ==================================================================
    def _show_page(self, name: str) -> None:
        self._page_area.clear_output(wait=True)
        with self._page_area:
            display(self._pages[name])

    def _on_nav(self, change) -> None:
        self._show_page(change["new"])

    # ==================================================================
    # 表示
    # ==================================================================
    def show(self) -> None:
        display(self.ui)

    # ==================================================================
    # ステータス
    # ==================================================================
    def _set_status(self, msg: str, color: str = "#333") -> None:
        self.w_status.value = f"<b style='color:{color};'>{msg}</b>"

    # ==================================================================
    # コールバック
    # ==================================================================
    def _on_train(self, _) -> None:
        self.out.clear_output()
        with self.out:
            vtu = self.w_train_file.value
            if not vtu:
                self._set_status("VTU ファイルを選択してください", "red"); return
            self._set_status("学習中…", "blue")
            self.w_progress.layout.visibility = "visible"
            self.w_progress.max = self.w_epochs.value
            self.w_progress.value = 0
            self.tk = GNNToolkit(
                data_dir=self.data_dir,
                results_dir=self.results_dir,
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

    def _on_predict(self, _) -> None:
        self.out.clear_output()
        with self.out:
            if self.tk is None or not self.tk.is_trained:
                self._set_status("先にモデルを学習または読込してください", "red"); return
            self._set_status("推論中…", "blue")
            res = self.tk.predict(self.w_pred_file.value, self.w_pred_load.value,
                                  self.w_pred_output.value.strip() or None)
            self._set_status(
                f"推論完了 — "
                f"X:{res['max_disp_x']:.4f} / "
                f"Y:{res['max_disp_y']:.4f} / "
                f"Z:{res['max_disp_z']:.4f} mm  |  "
                f"応力 {res['max_stress']:.3f} MPa", "green")
            self._refresh_vtu_lists()

    def _on_evaluate(self, _) -> None:
        self.out.clear_output()
        with self.out:
            if self.tk is None or not self.tk.is_trained:
                self._set_status("先にモデルを学習または読込してください", "red"); return
            self._set_status("評価中…", "blue")
            res = self.tk.evaluate(self.w_eval_file.value, self.w_eval_load.value)
            self._set_status(
                f"評価完了 — "
                f"変位誤差 X:{res.get('d_rel_x',0):.2f}% "
                f"Y:{res.get('d_rel_y',0):.2f}% "
                f"Z:{res.get('d_rel_z',0):.2f}%  |  "
                f"応力誤差 {res['s_rel']:.2f}%", "green")

    def _on_plot_loss(self, _) -> None:
        self.out.clear_output()
        with self.out:
            if self.tk is None or not self.tk.loss_history:
                self._set_status("学習履歴がありません", "red"); return
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.semilogy(self.tk.loss_history)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Total Loss")
            ax.set_title("学習 Loss 曲線"); ax.grid(True, alpha=0.3)
            plt.tight_layout(); plt.show()
            self._set_status("Loss 曲線を表示しました", "green")

    def _on_save(self, _) -> None:
        self.out.clear_output()
        with self.out:
            if self.tk is None or not self.tk.is_trained:
                self._set_status("保存するモデルがありません", "red"); return
            d = self.w_save_dir.value.strip()
            if not d:
                self._set_status("保存先を入力してください", "red"); return
            self.tk.save(d)
            self._set_status(f"モデルを {d}/ に保存しました", "green")
            self._refresh_model_dirs()

    def _on_load(self, _) -> None:
        self.out.clear_output()
        with self.out:
            d = self.w_load_dir.value
            if not d or d == "(なし)":
                self._set_status("読込元を選択してください", "red"); return
            if self.tk is None:
                self.tk = GNNToolkit(
                    data_dir=self.data_dir,
                    results_dir=self.results_dir,
                    train_load=self.w_train_load.value,
                )
            self.tk.load(d)
            self._set_status(f"モデルを {d}/ から読み込みました", "green")

    def _on_refresh(self, _) -> None:
        self._refresh_vtu_lists()
        self._refresh_model_dirs()
        self._set_status("ファイル一覧を更新しました", "#333")

    def _on_analyze(self, _) -> None:
        self.out.clear_output()
        with self.out:
            vtu = self.w_analyze_file.value
            if not vtu:
                self._set_status("VTU ファイルを選択してください", "red"); return
            from .data import FEADataProcessor
            FEADataProcessor.analyze(vtu)
            self._set_status(f"{vtu} の解析完了", "green")

    # ==================================================================
    # リフレッシュ
    # ==================================================================
    def _refresh_vtu_lists(self) -> None:
        vtu_list = _vtu_files(self.data_dir)
        for w in [self.w_train_file, self.w_pred_file, self.w_eval_file, self.w_analyze_file]:
            w.options = vtu_list

    def _refresh_model_dirs(self) -> None:
        dirs = _model_dirs(self.work_dir)
        self.w_load_dir.options = dirs if dirs else ["(なし)"]

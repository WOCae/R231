# 汎用構造解析 GNN ツールキット — 仕様書

| 項目       | 内容                           |
|-----------|-------------------------------|
| バージョン | 3.1.0                         |
| 作成日     | 2026-02-17                    |
| 言語       | Python 3.10+                  |
| フレームワーク | PyTorch / PyTorch Geometric |

---

## 1. 概要

本ツールキットは、有限要素解析（FEA）の結果ファイル（VTU形式）からグラフニューラルネットワーク（GNN）を自動構築・学習し、**任意の荷重条件・荷重方向**および**異なるメッシュ形状**での応力・変位を高速に推論するソフトウェアである。

GUI（ipywidgets）と Python API の2つのインターフェースを提供し、Jupyter Notebook 上での直観的な操作と、スクリプトからの自動化の両方に対応する。

### v3 新機能: メッシュ形状変更対応

| 推論パターン | 説明 | 必要データ |
|-------------|------|----------|
| **荷重スケーリング** | 同メッシュ・異荷重 | 1ファイル以上 |
| **荷重方向変更** | 同メッシュ・引張+曲げ等 | 各方向のVTU |
| **形状汎化** | 異メッシュ・任意荷重 | 複数形状ファイル推奨 |

---

## 2. ディレクトリ構成

```
R231/
├── R231.ipynb              # メインノートブック（GUI / コード実行）
├── gnn_toolkit/            # Pythonパッケージ
│   ├── __init__.py         # パッケージエントリポイント
│   ├── config.py           # GNNConfig — 設定・自動キャリブレーション
│   ├── data.py             # FEADataProcessor — VTU解析・グラフ変換
│   ├── inp_reader.py       # InpFileReader — CalculiX .inp パーサー
│   ├── model.py            # StructuralGNN — GNNモデル定義
│   ├── toolkit.py          # GNNToolkit — ファサード（学習・推論・評価・保存）
│   └── ui.py               # GNNToolkitUI — ipywidgets GUI
├── data/                   # 学習用 VTU ファイル
├── results/                # 推論結果 VTU ファイル
├── *.vtu                   # FEA解析結果ファイル
└── saved_model/            # 学習済みモデル保存先
    ├── model.pth
    └── config.json
```

---

## 3. モジュール仕様

### 3.1 GNNConfig（config.py）

設定パラメータを一元管理するデータクラス。

#### パラメータ一覧

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `to_mm` | float | 1000.0 | m → mm 変換係数 |
| `to_mpa` | float | 1e-6 | Pa → MPa 変換係数 |
| `norm_coord` | float? | None（自動） | 座標正規化定数 |
| `norm_disp` | float? | None（自動） | 変位正規化定数 [mm] |
| `norm_stress` | float? | None（自動） | 応力正規化定数 [MPa] |
| `train_load` | float | 1000.0 | 基準荷重 [N] |
| `hidden_dim` | int | 128 | 隠れ層の次元数 |
| `n_layers` | int | 4 | GNN 層数 |
| `n_features` | int | 14 | 入力特徴量数（14:ジオメトリ / 8:レガシー） |
| `n_outputs` | int | 4 | 出力次元数 (ux,uy,uz,stress) |
| `include_geometry` | bool | True | ジオメトリ特徴量の使用（形状汎化モード） |
| `epochs` | int | 5000 | 最大学習エポック数 |
| `lr` | float | 0.001 | 初期学習率 |
| `lr_min` | float | 1e-5 | Cosine Annealing 最小学習率 |
| `stress_weight` | float | 100.0 | 応力損失の重み係数 |
| `patience` | int | 500 | Early Stopping の猶予エポック数 |
| `load_ratios` | list[float] | [0.1..1.0] | データ拡張用荷重倍率 |
| `linear_scaling` | bool | True | 線形弾性スケーリングの有効化 |
| `disp_key` | str? | None（自動） | 変位データのキー名 |
| `stress_key` | str? | None（自動） | 応力データのキー名 |

#### メソッド

| メソッド | 引数 | 戻り値 | 説明 |
|---|---|---|---|
| `auto_calibrate` | `mesh`, `load_N?` | None | メッシュから正規化定数を自動計算 |
| `auto_calibrate_multi` | `meshes`, `load_values` | None | 複数メッシュから正規化定数を自動計算 |
| `update_n_features` | — | None | include_geometry に応じて n_features を設定 |
| `save` | `path: str` | None | JSON に設定を保存 |
| `load` | `path: str` | GNNConfig | JSON から設定を復元（クラスメソッド） |
| `summary` | — | str | 設定の要約文字列 |

#### キー自動検出の優先順位

**変位キー**:
1. `displacement`, `disp` を含むもの
2. `u_`, `uvw` を含むもの

**応力キー**:
1. `mises`, `von` を含むもの
2. `eqv`, `equivalent` を含むもの
3. `stress` を含むもの

---

### 3.2 FEADataProcessor（data.py）

VTU ファイルの読込・解析・境界条件検出・PyG グラフ変換を行うユーティリティクラス。

#### メソッド

| メソッド | 引数 | 戻り値 | 説明 |
|---|---|---|---|
| `analyze` | `path: str` | UnstructuredGrid | VTU の構造を解析・表示 |
| `detect_bc` | `mesh`, `disp_data?` | `(is_fixed, is_load)` | 境界条件の自動検出 |
| `to_graph` | `vtu_path`, `load_N`, `config`, `load_direction?` | `Data` | VTU → PyG Data 変換 |
| `to_graph_from_inp` | `inp_path`, `load_N?`, `config`, `load_direction?` | `(Data, InpFileReader)` | INP → PyG Data 変換 |
| `detect_load_direction` | `mesh`, `disp_data?` | ndarray(3) | 荷重方向の自動検出 |
| `compute_geometry_features` | `pos`, `edge_index` | ndarray(N,9) | ジオメトリ特徴量計算 |
| `list_vtu_files` | `directory?` | list[str] | ディレクトリ内の VTU 一覧 |

#### 境界条件検出アルゴリズム

```
IF 変位データが存在:
    拘束ノード = 変位振幅 ≤ (最大変位 × 1e-6)
    荷重ノード = 最大変位ノードの属する端面
ELSE:
    拘束ノード = 最長軸の min 端
    荷重ノード = 最長軸の max 端
```

#### ノード特徴量（ジオメトリモード: 14次元, `include_geometry=True`）

| Index | 内容 | 正規化 |
|---|---|---|
| 0–2 | 座標 (x, y, z) | bbox → [0, 1] |
| 3 | 接続次数 | ÷ 最大次数 |
| 4 | 平均エッジ長 | ÷ 最大平均エッジ長 |
| 5 | エッジ長標準偏差 | ÷ 最大標準偏差 |
| 6–8 | 擬似法線（隣接ベクトル平均方向） | 単位ベクトル |
| 9 | 拘束フラグ | 0 or 1 |
| 10 | 荷重面フラグ | 0 or 1 |
| 11 | 荷重Fx | is_load × direction[0] × (load_N/train_load) |
| 12 | 荷重Fy | is_load × direction[1] × (load_N/train_load) |
| 13 | 荷重Fz | is_load × direction[2] × (load_N/train_load) |

荷重方向ベクトル (Fx, Fy, Fz) により、引張・曲げ・任意方向の荷重条件を学習・推論できる。

#### ノード特徴量（レガシーモード: 8次元, `include_geometry=False`）

| Index | 内容 | 正規化 |
|---|---|---|
| 0–2 | 座標 (x, y, z) | ÷ norm_coord |
| 3 | 拘束フラグ | 0 or 1 |
| 4 | 荷重面フラグ | 0 or 1 |
| 5 | 荷重Fx | is_load × direction[0] × (load_N/train_load) |
| 6 | 荷重Fy | is_load × direction[1] × (load_N/train_load) |
| 7 | 荷重Fz | is_load × direction[2] × (load_N/train_load) |

レガシーモードは v2 以前との後方互換。同メッシュ・荷重スケーリングのみ対応。
荷重方向ベクトルは v3.1 で追加。

---

### 3.3 InpFileReader（inp_reader.py）

CalculiX .inp ファイルを解析し、メッシュ・境界条件・荷重情報を取得するパーサー。
推論時に `.inp` を指定することで、拘束・荷重を明示的に伝えられる。

#### 対応キーワード

| キーワード | 説明 |
|---|---|
| `*NODE` | ノード座標 (ID, x, y, z) |
| `*ELEMENT` | 要素接続 (C3D4, C3D10, C3D8) |
| `*NSET` | ノード集合（通常 / GENERATE） |
| `*BOUNDARY` | 拘束条件（NSET名 または ノードID） |
| `*CLOAD` | 集中荷重（NSET名 または ノードID） |

#### 対応要素タイプ

| 要素 | 節点数 | VTK タイプ | エッジ数/要素 |
|---|---|---|---|
| C3D4 | 4 | TETRA (10) | 6 |
| C3D10 | 10 | QUADRATIC_TETRA (24) | 12 |
| C3D8 | 8 | HEXAHEDRON (12) | 12 |

#### メソッド

| メソッド | 戻り値 | 説明 |
|---|---|---|
| `read(path)` | self | .inp ファイルを解析 |
| `get_positions()` | ndarray(N,3) | ノード座標 |
| `get_edges()` | ndarray(E,2) | エッジペア（0-based） |
| `get_fixed_mask()` | ndarray(N) | 拘束マスク (0/1) |
| `get_load_mask()` | ndarray(N) | 荷重マスク (0/1) |
| `get_total_load()` | float | CLOAD 合力ノルム [N] |
| `get_load_direction()` | ndarray(3) | CLOAD 荷重方向単位ベクトル |
| `to_pyvista()` | UnstructuredGrid | PyVista メッシュ構築 |
| `summary()` | str | 解析サマリー |

---

### 3.4 StructuralGNN（model.py）

グラフニューラルネットワーク本体。Encoder–Processor–Decoder アーキテクチャ。

#### アーキテクチャ図

```
入力 (14次元 [ジオメトリ] / 8次元 [レガシー])
  ↓
[Encoder] Linear → ReLU → Linear     → 潜在空間 (hidden_dim)
  ↓
[Processor] _ResBlock × n_layers
  │  SAGEConv → LayerNorm → ReLU + Residual
  ↓
[Decoder]
  ├→ disp_head:   Linear→ReLU→Linear→ReLU→Linear → 3次元 (ux,uy,uz)
  └→ stress_head: Linear→ReLU→Linear→ReLU→Linear → 1次元 (σ_mises)
  ↓
  × ||load_vec||  (linear_scaling=True の場合、荷重ベクトルのノルム)
  ↓
出力 (4次元)
```

#### パラメータ数の目安

| hidden_dim | n_layers | パラメータ数 |
|---|---|---|
| 64 | 2 | ~25,000 |
| 128 | 4 | ~200,000 |
| 256 | 6 | ~800,000 |

---

### 3.5 GNNToolkit（toolkit.py）

学習・推論・評価・保存/読込を統合するファサードクラス。

#### コンストラクタ

```python
GNNToolkit(**kwargs)
```
`kwargs` は `GNNConfig` のパラメータをそのまま受け取る。

#### メソッド

| メソッド | 引数 | 戻り値 | 説明 |
|---|---|---|---|
| `train` | `vtu_files`, `load_values?`, `load_directions?`, `callback?` | list[float] | GNN モデルを学習。Loss 履歴を返す |
| `predict` | `mesh_file`, `load_N?`, `output_vtu?`, `load_direction?` | dict | 推論結果を VTU に保存（VTU / .inp 両対応） |
| `evaluate` | `vtu_file`, `load_N?` | dict | 正解との誤差を算出 |
| `save` | `directory: str` | None | モデルと設定を保存 |
| `load` | `directory: str` | None | 保存済みモデルを読込 |

#### `train()` の学習フロー

```
1. VTU読込 → auto_calibrate
2. StructuralGNN 構築
3. VTU → PyG Data 変換
4. 学習ループ:
   ├ データ拡張 (load_ratios × 6パターン)
   ├ 損失 = MSE(変位) + stress_weight × MSE(応力)
   ├ 勾配クリッピング (max_norm=1.0)
   ├ Cosine Annealing LR スケジューラ
   └ Early Stopping (patience 超過 & epoch>1000)
```

#### `predict()` の戻り値

```python
{
    "max_disp":   float,   # 最大変位（全方向）[mm]
    "max_disp_x": float,   # 最大変位 X方向 [mm]
    "max_disp_y": float,   # 最大変位 Y方向 [mm]
    "max_disp_z": float,   # 最大変位 Z方向 [mm]
    "max_stress": float,   # 最大応力 [MPa]
    "output":     str,     # 出力 VTU パス
}
```

#### VTU 出力フィールド

| フィールド | 型 | 説明 |
|---|---|---|
| `gnn_disp` | float[N,3] | 3方向変位ベクトル [mm] |
| `gnn_disp_x` | float[N] | X方向変位 [mm] |
| `gnn_disp_y` | float[N] | Y方向変位 [mm] |
| `gnn_disp_z` | float[N] | Z方向変位 [mm] |
| `gnn_stress` | float[N] | von Mises 応力 [MPa] |

#### `evaluate()` の戻り値

```python
{
    "d_mae": float,     # 変位 平均絶対誤差（全方向）[mm]
    "d_max": float,     # 変位 最大誤差（全方向）[mm]
    "d_rel": float,     # 変位 相対誤差（全方向）[%]
    "d_mae_x": float,   # 変位X 平均絶対誤差 [mm]
    "d_max_x": float,   # 変位X 最大誤差 [mm]
    "d_rel_x": float,   # 変位X 相対誤差 [%]
    "d_mae_y": float,   # 変位Y 平均絶対誤差 [mm]
    "d_max_y": float,   # 変位Y 最大誤差 [mm]
    "d_rel_y": float,   # 変位Y 相対誤差 [%]
    "d_mae_z": float,   # 変位Z 平均絶対誤差 [mm]
    "d_max_z": float,   # 変位Z 最大誤差 [mm]
    "d_rel_z": float,   # 変位Z 相対誤差 [%]
    "s_mae": float,     # 応力 平均絶対誤差 [MPa]
    "s_max": float,     # 応力 最大誤差 [MPa]
    "s_rel": float,     # 応力 相対誤差 [%]
}
```

---

### 3.6 GNNToolkitUI（ui.py）

ipywidgets ベースの対話型 GUI。5つのタブで全操作をカバーする。

#### タブ構成

| タブ | 機能 | 操作項目 |
|---|---|---|
| **学習** | GNN モデルの学習 | VTU選択、基準荷重、Epochs、Hidden、層数、応力重み、Patience、学習率、線形スケーリング |
| **推論** | 任意荷重での推論 | VTU/INP選択、荷重値、荷重方向、出力ファイル名 |
| **評価** | 精度評価・Loss曲線 | VTU選択、荷重値、精度評価ボタン、Loss曲線ボタン |
| **保存/読込** | モデルの永続化 | 保存先入力、読込元選択、ファイル更新 |
| **VTU解析** | VTU ファイルの中身確認 | VTU選択、解析ボタン |

#### ステータスバー

操作の結果をリアルタイムで表示:
- 青: 処理中
- 緑: 成功
- 赤: エラー

#### ログ出力

各操作の詳細ログがスクロール可能な出力領域に表示される。

---

## 4. 使用方法

### 4.1 GUI モード

```python
from gnn_toolkit import GNNToolkitUI

ui = GNNToolkitUI()
ui.show()
```

### 4.2 コードモード（基本）

```python
from gnn_toolkit import GNNToolkit

# include_geometry=True（デフォルト）で形状汎化モード
tk = GNNToolkit(train_load=1000.0)
tk.train("result_1000N.vtu")
tk.evaluate("result_1000N.vtu")
tk.predict("result_1000N.vtu", load_N=500.0)
tk.save("my_model")
```

### 4.3 コードモード（カスタム設定）

```python
tk = GNNToolkit(
    train_load=1000.0,
    epochs=8000,
    hidden_dim=256,
    n_layers=6,
    stress_weight=200.0,
    lr=0.0005,
    patience=800,
)
```

### 4.4 複数ファイル学習（同メッシュ・異荷重）

```python
tk = GNNToolkit(train_load=1000.0)
tk.train(
    ["result_500N.vtu", "result_1000N.vtu"],
    load_values=[500.0, 1000.0]
)
```

### 4.5 形状汎化学習（複数メッシュ）

```python
tk = GNNToolkit(include_geometry=True)  # ジオメトリ特徴量 ON
tk.train(
    ["mesh_A.vtu", "mesh_B.vtu"],
    load_values=[1000, 500]
)
# 未知メッシュへの推論
tk.predict("mesh_C.vtu", load_N=750.0)
```

### 4.6 荷重方向汎化学習（引張 + 曲げ）

```python
# 引張と曲げの2種のVTUで学習
tk = GNNToolkit(train_load=1000.0)
tk.train(
    ["tension_1000N.vtu", "bending_1000N.vtu"],
    load_values=[1000, 1000],
)  # 荷重方向は変位から自動検出

# .inp で任意方向の推論（CLOADから荷重方向自動取得）
tk.predict("FEMMeshNetgen.inp")
```

### 4.7 CalculiX .inp ファイルから推論

```python
# 拘束・荷重を .inp から明示的に読み取り
tk.predict("FEMMeshNetgen.inp")              # 荷重は CLOAD から自動計算
tk.predict("FEMMeshNetgen.inp", load_N=2000) # 荷重を変更して推論
```

### 4.8 保存済みモデルの再利用

```python
tk = GNNToolkit()
tk.load("saved_model")
tk.predict("mesh.vtu", load_N=1200.0)
```

---

## 5. 物理モデル

### 5.1 線形弾性スケーリング

`linear_scaling=True` の場合、モデルは**基準荷重（1000N）時点の単位応答**を学習し、推論時に荷重倍率を乗算する。

$$
\mathbf{u}(F) = \mathbf{u}_{base} \times \frac{F}{F_{train}}
$$
$$
\sigma(F) = \sigma_{base} \times \frac{F}{F_{train}}
$$

この制約により、線形弾性範囲内での外挿精度が大幅に向上する。

### 5.2 データ拡張

学習時に `load_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]` の6パターンで荷重倍率を変化させ、単一VTUから6倍のデータを生成する。

---

## 6. 依存パッケージ

| パッケージ | 最低バージョン | 用途 |
|---|---|---|
| `torch` | 2.0+ | ニューラルネットワーク |
| `torch_geometric` | 2.4+ | グラフニューラルネットワーク |
| `pyvista` | 0.40+ | VTU ファイル読み書き |
| `numpy` | 1.24+ | 数値計算 |
| `ipywidgets` | 8.0+ | GUI |
| `matplotlib` | 3.7+ | Loss 曲線プロット |

---

## 7. 入出力仕様

### 入力

| 項目 | 形式 | 説明 |
|---|---|---|
| 学習データ | `.vtu` (VTK Unstructured Grid) | FEA 解析結果（変位・応力含む） |
| 推論入力 | `.vtu` または `.inp` (CalculiX) | メッシュ形状と境界条件 |
| 荷重値 | float | 推論したい荷重条件 [N]（.inp は自動取得可） |

### 出力

| 項目 | 形式 | 説明 |
|---|---|---|
| 推論結果 | `.vtu` | `gnn_disp`（変位 [mm]）と `gnn_stress`（応力 [MPa]）を格納 |
| モデル | `model.pth` + `config.json` | 学習済み重みと設定 |

---

## 8. 制約事項・注意

1. **線形弾性**を前提とする（`linear_scaling=True`）。非線形材料・大変形には `linear_scaling=False` に変更し、複数荷重のVTUを用意する必要がある。
2. VTU に**変位**と**応力**のデータが含まれている必要がある。
3. `include_geometry=False`（レガシーモード）では入力メッシュのトポロジー（ノード数・接続）が推論時と学習時で同一であること。ジオメトリモード (`True`) では異なるメッシュでも推論可能。
4. 形状汎化精度は、学習データに含まれる形状バリエーションに依存する。類似カテゴリの形状で学習するほど精度が向上する。
5. GPU 利用時は CUDA 対応の PyTorch がインストールされていること。

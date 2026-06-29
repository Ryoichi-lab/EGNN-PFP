# tmQM_dipole

**対象物性**: 双極子モーメント μ (D)
**データセット**: tmQM（遷移金属錯体、約86,000錯体、TPSSh-D3BJ/def2-SVP、GFN2-xTB構造最適化）
**手法**: EGNN(baseline) vs EGNN×PFP記述子

## 結果サマリ

| 構成 | MAE (D) |
|---|---|
| baseline (EGNN) | 1.447 |
| PFP記述子付き (EGNN×PFP) | 0.540 |

出典: Uchiyama et al., arXiv:2602.03046, Table 4

## モデル設定の補足

- PFP記述子（256次元原子embedding）は Matlantis の **r2SCANモード**で計算。
- エッジ特徴量に **Wiberg bond order（GFN2-xTBで計算）** を追加。
- message aggregation は **mean**（配位数の偏りを避けるため）。
- coordinate update は **ON**（xTB最適化構造はDFT最適化構造より精度が低いため、学習中に構造を微調整する）。

## ディレクトリ構成

```
tmQM_dipole/
├── README.md
├── scripts/    # グラフ構築・学習(baseline/PFP/PFP+BO)・評価/比較プロット作成スクリプト
└── results/    # baseline vs PFP 比較プロット（yy-plot・誤差分布等）
```

`data/`・`descriptors/` ディレクトリは作成していません（xyz構造・PFP記述子は大容量のため、公開リポジトリには含めない方針）。

### scripts/ の内容

| ファイル | 役割 |
|---|---|
| `01_import_xyz.py` | tmQM公式xyzファイルから中性一重項構造を抽出 |
| `02_make_graph_dipole.py` | baseline/PFP両方のグラフ(PyG Data)を構築 |
| `03_train_egnn_pfp_dipole_v1.py` | EGNN×PFP 学習（PFP記述子のみ付加、BOなし版） |
| `04_train_egnn_baseline_dipole.py` | EGNN baseline 学習 |
| `05_import_bond_order.py` | GFN2-xTB Wiberg bond order ファイルの読み込み |
| `06_add_bond_order_to_graph_dipole.py` | PFPグラフに bond order エッジ特徴量を追加 |
| `07_train_egnn_pfp_bo_dipole.py` | EGNN×PFP+BO 学習（最終モデル） |
| `08_compare_plot_dipole.py` | baseline vs PFP+BO の比較プロット作成（results/の図を生成） |
| `check_number_of_atoms.py` | 256原子超の大規模分子の検出・抽出 |

グラフ構築・bond order関連スクリプトは他のtmQM物性（HOMO, LUMO, HOMO-LUMOギャップ, Metal_q）と共通の構造（同一スクリプト群を物性ごとにコピーして利用）です。

# tmQM_HOMO

**対象物性**: HOMOエネルギー εHOMO (eV)
**データセット**: tmQM（遷移金属錯体、約86,000錯体、TPSSh-D3BJ/def2-SVP、GFN2-xTB構造最適化）
**手法**: EGNN(baseline) vs EGNN×PFP記述子

## 結果サマリ

| 構成 | MAE (eV) |
|---|---|
| baseline (EGNN) | 0.147 |
| PFP記述子付き (EGNN×PFP) | 0.082 |

出典: Uchiyama et al., arXiv:2602.03046, Table 4

## モデル設定の補足

- PFP記述子（256次元原子embedding）は Matlantis の **r2SCANモード**で計算。
- エッジ特徴量に **Wiberg bond order（GFN2-xTBで計算）** を追加。
- message aggregation は **mean**（配位数の偏りを避けるため）。
- coordinate update は **ON**（xTB最適化構造はDFT最適化構造より精度が低いため、学習中に構造を微調整する）。

## ディレクトリ構成

```
tmQM_HOMO/
├── README.md
├── scripts/    # グラフ構築・学習(baseline/PFP/PFP+BO)・評価/比較プロット作成スクリプト
└── results/    # baseline vs PFP 比較プロット（yy-plot・誤差分布等）
```

`data/`・`descriptors/` ディレクトリは作成していません（xyz構造・PFP記述子は大容量のため、公開リポジトリには含めない方針）。

### scripts/ の内容

| ファイル | 役割 |
|---|---|
| `00_import_xyz.py` | tmQM公式xyzファイルから中性一重項構造を抽出（dipoleと共通） |
| `00_import_bond_order.py` | GFN2-xTB Wiberg bond order ファイルの読み込み（共通） |
| `00_check_number_of_atoms.py` | 256原子超の大規模分子の検出・抽出（共通） |
| `01_make_graph_homo.py` | baseline/PFP両方のグラフ(PyG Data)を構築 |
| `02_train_egnn_pfp_homo.py` | EGNN×PFP 学習（PFP記述子のみ付加、BOなし版） |
| `03_train_egnn_baseline_homo.py` | EGNN baseline 学習 |
| `04_add_bond_order_to_graph_homo.py` | PFPグラフに bond order エッジ特徴量を追加 |
| `05_train_egnn_pfp_bo_homo.py` | EGNN×PFP+BO 学習（最終モデル） |
| `06_compare_baseline_vs_pfp_homo.py` | baseline/PFPモデルの誤差分布比較 |
| `07_evaluate_pfp_homo.py` | PFPモデルの性能評価 |
| `08_compare_plot_homo.py` | baseline vs PFP+BO の比較プロット作成（results/の図を生成） |

グラフ構築・学習スクリプトは他のtmQM物性(μ, Δε, εLUMO, Metal_q)と共通の構造（同一スクリプト群を物性ごとにコピーして利用）です。

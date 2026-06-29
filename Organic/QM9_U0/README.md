# QM9_U0

**対象物性**: 内部エネルギー 0K (U₀)
**データセット**: QM9
**手法**: EGNN(baseline) vs EGNN×PFP記述子

## 結果サマリ

| 構成 | MAE (meV) | R² |
|---|---|---|
| baseline (EGNN) | 13.6 | - |
| PFP記述子付き (EGNN×PFP) | 9.7 | - |

出典: Uchiyama et al., arXiv:2602.03046, Table 3

参考値（自前学習run、本リポジトリのscripts/で再現したもの）:

| 構成 | Val/Test MAE (meV) | R² |
|---|---|---|
| baseline (EGNN) | 14.0 (val) | - |
| PFP記述子付き (EGNN×PFP) | 9.67 (test) | 0.999997 |

## ディレクトリ構成

```
QM9_U0/
├── scripts/
│   ├── 1_import_data.py    # QM9から原子化エネルギー(U0)を抽出・補正
│   ├── 2_build_graph.py    # xyz+PFP記述子からグラフ(baseline/PFP両方)を構築
│   ├── 3_train_baseline.py # EGNN baseline学習
│   ├── 4_train_pfp.py      # EGNN×PFP学習
│   └── 5_make_plots.py     # テスト評価・散布図/誤差分析プロット作成
└── results/
    ├── baseline_predictions.png
    ├── pfp_predictions.png
    └── pfp_test_comprehensive_analysis.png
```

元データ・PFP記述子はサイズの都合上ローカルのみで管理しています（本リポジトリにはコピーしていません）。

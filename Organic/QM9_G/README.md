# QM9_G

**対象物性**: 自由エネルギー (G)
**データセット**: QM9
**手法**: EGNN(baseline) vs EGNN×PFP記述子

## 結果サマリ

| 構成 | MAE (meV) | R² |
|---|---|---|
| baseline (EGNN) | 10.8 | - |
| PFP記述子付き (EGNN×PFP) | 10.5 | - |

出典: Uchiyama et al., arXiv:2602.03046, Table 3

参考値（自前学習run・本リポジトリ移植元の training_stats_*.json より）:

| 構成 | best_val_mae (meV) |
|---|---|
| baseline (training_baseline_G_egnn_1217) | 19.15 |
| PFP記述子付き (QM9_G_training_pfp_egnn_1218) | 15.47 |

※自前学習runは論文公式数値とは別条件（エポック数・乱数シード等）での実行結果であり、論文の数値と直接比較できる値ではない。

## ディレクトリ構成

```
QM9_G/
├── scripts/
│   ├── 1_import_data.py      # QM9からGラベルを抽出・Hartree変換 (1.import_data.py)
│   ├── 2_build_graph.py      # XYZ+PFP記述子からグラフ構築 (2.make_graph.py)
│   ├── 3_train_baseline.py   # EGNN baseline学習 (4.1.make_EGNN_base_2.py)
│   └── 4_train_pfp.py        # EGNN×PFP学習 (3.2.make_EGNN_pfp_coordupdate.py)
└── results/
    ├── baseline_training_history.png
    └── pfp_training_history.png
```

元データ・PFP記述子はサイズの都合上ローカルのみで管理（本リポジトリには含まない）。

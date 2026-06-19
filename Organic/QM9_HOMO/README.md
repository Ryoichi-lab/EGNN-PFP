# QM9_HOMO

**対象物性**: HOMOエネルギー
**データセット**: QM9
**手法**: EGNN（baseline） vs EGNN×PFP記述子

## 結果サマリ

| 構成 | MAE (meV) | R² |
|---|---|---|
| baseline (EGNN) | 28.2 | - |
| PFP記述子付き (EGNN×PFP) | 22.4 | - |

出典: Uchiyama et al., arXiv:2602.03046, Table 3

参考値（自前学習run）: baseline run (`training_baseline_HOMO_egnn_`) の best_val_mae = 47.3 meV、PFP run (`training_pfp_HOMO_egnn_official_128_attention_perfect_0110`) の best_val_mae = 22.5 meV（training_stats_*.jsonより。R²の記録なし）。

## ディレクトリ構成

```
QM9_HOMO/
├── scripts/
│   ├── 1_import_data.py    # QM9からHOMOエネルギー・座標を抽出しxyz化
│   ├── 2_build_graph.py    # xyz + PFP記述子からグラフ(.pt)を構築（perfect版）
│   ├── 3_train_baseline.py # EGNN baseline学習（PFPなし）
│   └── 4_train_pfp.py      # EGNN×PFP学習
└── results/
    ├── baseline_training_history.png
    ├── pfp_training_history.png
    └── pfp_predictions_scatter.png
```

元データ・PFP記述子はサイズの都合上ローカルのみで管理（本リポジトリには含まない）。

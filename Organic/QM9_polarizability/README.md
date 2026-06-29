# QM9_polarizability

**対象物性**: 等方分極率 (α)
**データセット**: QM9
**手法**: EGNN(baseline) vs EGNN×PFP記述子

## 結果サマリ

| 構成 | MAE (Bohr³) | R² |
|---|---|---|
| baseline (EGNN) | 0.061 | - |
| PFP記述子付き (EGNN×PFP) | 0.060 | - |

出典: Uchiyama et al., arXiv:2602.03046, Table 3

R²は論文には記載がなく、移植元の自前学習run(baseline: `training_baseline_polarizability_perfect_0115`, best_val_mae=0.0607; PFP: `training_pfp_polarizability_egnn_official_128_attention_perfect_0110`, best_val_mae=0.0596)の `training_results.json` にもR²の記録がないため省略。

## ディレクトリ構成

```
QM9_polarizability/
├── scripts/
│   ├── 1_import_data.py    # QM9から分極率・座標を抽出しxyz化
│   ├── 2_build_graph.py    # xyz+PFP記述子からグラフ構築（baseline/pfp両方を出力）
│   ├── 3_train_baseline.py # EGNN baseline学習
│   └── 4_train_pfp.py      # EGNN×PFP学習
└── results/
    ├── baseline_training_history.png
    ├── baseline_predictions.png
    ├── pfp_training_history.png
    └── pfp_predictions.png
```

元データ・PFP記述子はサイズの都合上ローカルのみで管理。

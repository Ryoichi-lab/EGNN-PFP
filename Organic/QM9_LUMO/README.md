# QM9_LUMO

**対象物性**: LUMOエネルギー
**データセット**: QM9
**手法**: EGNN（baseline） vs EGNN×PFP記述子

## 結果サマリ

| 構成 | MAE (meV) | R² |
|---|---|---|
| baseline (EGNN) | 23 | - |
| PFP記述子付き (EGNN×PFP) | 21.0 | - |

出典: Uchiyama et al., arXiv:2602.03046, Table 3

（R²は元の training_results.json には記録されておらず、論文中にも記載がないため省略）

## ディレクトリ構成

```
QM9_LUMO/
├── scripts/
│   ├── 1_import_data.py    # QM9からLUMOエネルギー・構造をxyz抽出
│   ├── 2_build_graph.py    # xyz+PFP記述子からグラフ構築（baseline/PFP両対応）
│   ├── 3_train_baseline.py # EGNN baseline学習（PFPなし）
│   └── 4_train_pfp.py      # EGNN×PFP学習
└── results/
    ├── baseline_training_history.png
    ├── baseline_predictions.png
    ├── pfp_training_history.png
    └── pfp_predictions.png
```

元データ・PFP記述子はサイズの都合上ローカルのみで管理。

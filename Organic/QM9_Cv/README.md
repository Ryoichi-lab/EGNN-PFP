# QM9_Cv

**対象物性**: 定積熱容量 (Cv)
**データセット**: QM9
**手法**: EGNN（baseline） vs EGNN×PFP記述子

## 結果サマリ

| 構成 | MAE (cal/mol・K) | R² |
|---|---|---|
| baseline (EGNN) | 0.033 | - |
| PFP記述子付き (EGNN×PFP) | 0.027 | - |

出典: Uchiyama et al., arXiv:2602.03046, Table 3

参考値（自前学習run, `training_baseline_Cv_egnn_1216` / `training_pfp_Cv_egnn_1216`）: baseline best_val_MAE = 0.0384, EGNN-PFP best_val_MAE = 0.0281 cal/(mol・K)（training_stats_*.json記載。R²は記録されていない）

## ディレクトリ構成

```
QM9_Cv/
├── scripts/
│   ├── 1_import_data.py   # QM9からCvを抽出・原子参照エネルギー補正、xyz出力
│   ├── 2_build_graph.py   # xyz + PFP記述子からグラフ(.pt)構築（baseline/pfp両対応）
│   ├── 3_train_baseline.py # EGNN baseline学習（PFPなし）
│   └── 4_train_pfp.py      # EGNN×PFP学習
└── results/
    ├── baseline_training_history.png  # baseline学習曲線
    └── pfp_training_history.png       # EGNN-PFP学習曲線
```

元データ・PFP記述子・グラフファイル（.pt）・チェックポイント（.pth）はサイズの都合上ローカルのみで管理し、本リポジトリには含めていない。

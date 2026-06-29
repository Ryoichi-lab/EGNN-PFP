# QM9_H

**対象物性**: エンタルピー (H)
**データセット**: QM9
**手法**: EGNN(baseline) vs EGNN×PFP記述子

## 結果サマリ

| 構成 | MAE (meV) | R² |
|---|---|---|
| baseline (EGNN) | 11.3 | - |
| PFP記述子付き (EGNN×PFP) | 10.5 | - |

出典: Uchiyama et al., arXiv:2602.03046, Table 3

参考値（自前学習run、training_stats_*.json記載値）:

| 構成 | Val MAE (meV) | エポック数 |
|---|---|---|
| baseline (training_baseline_H_egnn_1217) | 37.9 | 298 |
| PFP記述子付き (QM9_H_training_pfp_egnn_1217) | 19.6 | 443 |

※ 上記の自前学習runは論文公式数値(Table 3)とは別の中間チェックポイントであり、論文の最終結果はTable 3の値を参照のこと。R²は論文に記載がないため省略。

## ディレクトリ構成

```
QM9_H/
├── scripts/
│   ├── 1_import_data.py    # QM9からHエンタルピーデータを抽出・変換
│   ├── 2_build_graph.py    # 分子グラフ構築（PFP記述子付与, perfect版）
│   ├── 3_train_baseline.py # EGNN baseline学習（PFPなし）
│   └── 4_train_pfp.py      # EGNN×PFP学習
└── results/
    ├── baseline_training_history.png  # baseline学習曲線
    └── pfp_training_history.png       # EGNN×PFP学習曲線
```

元データ・PFP記述子はサイズの都合上ローカルのみで管理（このリポジトリには含めていない）。

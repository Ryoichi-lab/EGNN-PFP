# QM9_HOMO_LUMO_gap

**対象物性**: HOMO-LUMOギャップ
**データセット**: QM9
**手法**: EGNN(baseline) vs EGNN×PFP記述子

## 結果サマリ

| 構成 | MAE (meV) | R² |
|---|---|---|
| baseline (EGNN) | 47.6 | - |
| PFP記述子付き (EGNN×PFP) | 40.4 | - |

出典: Uchiyama et al., arXiv:2602.03046, Table 3

R²は論文・自前学習run(`training_baseline_HOMO_LUMO_perfect_0113`, `training_pfp_gap_egnn_official_128_perfect_attention_0110`)のtraining_results.jsonいずれにも記載がないため省略。

## ディレクトリ構成

```
QM9_HOMO_LUMO_gap/
├── scripts/
│   ├── 1_import_data.py    # QM9からHOMO-LUMOギャップ・座標を抽出し.xyz化
│   ├── 2_build_graph.py    # .xyz + PFP記述子からグラフ構築（baseline/pfp両方）
│   ├── 3_train_baseline.py # EGNN baseline学習（PFPなし）
│   └── 4_train_pfp.py      # EGNN×PFP学習
└── results/
    ├── baseline_training_history.png
    ├── baseline_predictions.png
    ├── pfp_training_history.png
    └── pfp_predictions.png
```

元データ・PFP記述子はサイズの都合上ローカルのみで管理（本リポジトリには含めない）。

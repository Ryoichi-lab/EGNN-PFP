# QM9_ZPVE

**対象物性**: 零点振動エネルギー (ZPVE)
**データセット**: QM9
**手法**: EGNN(baseline) vs EGNN×PFP記述子

## 結果サマリ

| 構成 | MAE (meV) |
|---|---|
| baseline (EGNN) | 1.58 |
| PFP記述子付き (EGNN×PFP) | 1.33 |

出典: Uchiyama et al., arXiv:2602.03046, Table 3

参考値（自前学習run）: baseline run (`training_baseline_zpve_from_pfp_0115`) の best_val_mae = 1.5848 meV、PFP run (`training_pfp_zpve_egnn_official_128_attention_perfect_0112`) の best_val_mae = 1.3321 meV。R²は記録されていないため省略。

## ディレクトリ構成

```
QM9_ZPVE/
├── scripts/
│   ├── 1_import_data.py   # QM9からZPVEを取り出しxyz化
│   ├── 2_build_graph.py   # xyz+PFP記述子からグラフ構築（baseline/pfp両方を出力）
│   ├── 3_train_baseline.py # EGNN baseline学習（PFPなし）
│   ├── 4_train_pfp.py      # EGNN×PFP学習
│   └── 5_make_plots.py     # 学習済みモデルの評価・プロット作成
└── results/
    ├── baseline_training_history.png
    ├── baseline_predictions.png
    ├── pfp_training_history.png
    └── pfp_predictions.png
```

元データ・PFP記述子はサイズの都合上ローカルのみで管理（このリポジトリには含めていません）。

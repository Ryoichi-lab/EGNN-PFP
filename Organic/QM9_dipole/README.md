# QM9_dipole

**対象物性**: 双極子モーメント (μ)
**データセット**: QM9
**手法**: EGNN（baseline） vs EGNN×PFP記述子

## 結果サマリ

| 構成 | MAE (D) | R² |
|---|---|---|
| baseline (EGNN) | 0.029 | 参考値（自前学習run）: best_val_mae=0.0288 |
| PFP記述子付き (EGNN×PFP) | 0.022 | 参考値（自前学習run）: best_val_mae=0.0218 |

出典: Uchiyama et al., arXiv:2602.03046, Table 3

参考値は scripts/3_train_baseline.py / 4_train_pfp.py による自前学習run（training_baseline_dipole_B3LYP_0115, training_pfp_B3LYP_egnn_official_128_attention_perfect_0115）の training_results.json から。論文本文にはR²の記載はないため省略。

## ディレクトリ構成

```
QM9_dipole/
├── scripts/
│   ├── 1_import_data.py   # qm9_xyz の原子数行修復（ASE互換xyz生成）
│   ├── 2_build_graph.py   # 完全グラフ構築（baseline / PFP記述子付きグラフを生成）
│   ├── 3_train_baseline.py# EGNN baseline 学習
│   ├── 4_train_pfp.py     # EGNN×PFP 学習
│   └── 5_make_plots.py    # baseline vs PFP のテスト誤差比較・プロット作成
└── results/
    ├── baseline_training_history.png
    ├── baseline_predictions.png
    ├── pfp_training_history.png
    └── pfp_predictions.png
```

元データ・PFP記述子はサイズの都合上ローカルのみで管理（リポジトリには含めない）。

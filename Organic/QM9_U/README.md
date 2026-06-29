# QM9_U

**対象物性**: 内部エネルギー 298.15K (U)
**データセット**: QM9
**手法**: EGNN(baseline) vs EGNN×PFP記述子

## 結果サマリ

| 構成 | MAE (meV) | R² |
|---|---|---|
| baseline (EGNN) | 12.9 | - |
| PFP記述子付き (EGNN×PFP) | 10.1 | - |

出典: Uchiyama et al., arXiv:2602.03046, Table 3

参考値（自前学習run, test_evaluation/test_results.json より）: EGNN×PFP run(`QM9_u_training_pfp_egnn_128_attention_perfect_0110`)の R² = 0.999998（テストセット, N=12985）。MAE = 10.14 meV で論文値とほぼ一致。

## ディレクトリ構成

```
QM9_U/
├── README.md
├── scripts/
│   ├── 1_import_data.py    # QM9からU(298.15K)を抽出・原子参照エネルギー補正
│   ├── 2_build_graph.py    # xyz+PFP記述子からグラフ(.pt)構築（perfect版）
│   ├── 3_train_baseline.py # EGNN baseline学習（official実装）
│   ├── 4_train_pfp.py      # EGNN×PFP学習（attention付き）
│   ├── 5_evaluate_pfp.py   # PFPモデルのテスト評価（MAE/R²/散布図等）
│   └── 6_make_plots.py     # チェックポイントから学習履歴プロット再作成
└── results/
    ├── baseline_training_history.png       # baseline学習曲線
    ├── pfp_predictions_epoch500.png        # PFPモデル 予測 vs 実測 散布図
    └── pfp_test_comprehensive_analysis.png # PFPモデル テスト総合解析
```

元データ・PFP記述子はサイズの都合上ローカルのみで管理（本リポジトリには含めない）。学習済みチェックポイント(.pth)・グラフファイル(.pt)・ログ(.log)も同様の理由で除外。

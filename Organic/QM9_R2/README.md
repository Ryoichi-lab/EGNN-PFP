# QM9_R2

**対象物性**: 電子空間広がり ⟨R²⟩
**データセット**: QM9
**手法**: EGNN（baseline） vs EGNN×PFP記述子

## 結果サマリ

| 構成 | MAE (Bohr²) |
|---|---|
| baseline (EGNN) | 0.105 |
| PFP記述子付き (EGNN×PFP) | 0.168 |

出典: Uchiyama et al., arXiv:2602.03046, Table 3

**注意**: この物性は、本研究で扱った全物性の中で唯一PFP記述子の追加により予測精度が悪化する例外的なケースである。PFP記述子は局所的な電子状態の情報を捉えるのに有効だが、⟨R²⟩のような長距離・分子全体スケールの空間的広がりを表す物性には不向きであることを示す結果と考えられる。

なお、決定係数R²は論文には記載がないが、自前学習run（参考値）では以下が得られている。

| 構成 | best val MAE (参考値, 自前学習run) |
|---|---|
| baseline (`training_baseline_r2_from_pfp_0115`) | 0.0660 |
| PFP記述子付き (`training_pfp_r2_egnn_official_128_attention_perfect_0112`) | 0.1679 |

## ディレクトリ構成

```
QM9_R2/
├── scripts/
│   ├── 1_import_data.py   # QM9からR²・座標を抽出しxyzファイルとして保存
│   ├── 2_build_graph.py   # xyzファイル+PFP記述子からグラフ(.pt)を構築
│   ├── 3_train_baseline.py # EGNN baseline（PFPなし）学習スクリプト
│   └── 4_train_pfp.py     # EGNN×PFP学習スクリプト
└── results/
    ├── baseline_training_history.png
    ├── baseline_predictions.png
    ├── pfp_training_history.png
    └── pfp_predictions.png
```

元データ・PFP記述子・グラフファイル・チェックポイントはサイズの都合上ローカルのみで管理し、本リポジトリには含めていない。

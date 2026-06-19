# Metal_complex（遷移金属錯体 / tmQMg）

tmQMg データセット（遷移金属錯体、約3万分子）を対象に、EGNN×PFPによる物性予測を行います。

> **Note**: 遷移金属錯体ではNMRシールディングに相対論効果（ZORA等）による大きな補正が生じますが、
> 本フォルダは**まず非相対論ベースライン版**としてまとめています。相対論補正版（Scalar/SO-ZORA）は
> 別途進行中の発展版として今後追加予定です。

## 物性一覧

| フォルダ | 物性 | 説明 |
|---|---|---|
| [`tmQMg_dipole`](./tmQMg_dipole) | μ | 双極子モーメント |
| [`tmQMg_HOMO_LUMO_gap`](./tmQMg_HOMO_LUMO_gap) | gap | HOMO-LUMOギャップ |
| [`tmQMg_NMR_shielding`](./tmQMg_NMR_shielding) | σ（非相対論） | NMR等方シールディング（非相対論DFT） |

各フォルダ構成は概ね以下の通りです。

```
tmQMg_xxx/
├── README.md           # 物性の詳細・データ件数・結果サマリ
├── data/                # 構造・ラベル等
├── descriptors/         # PFP記述子
├── scripts/             # グラフ構築・学習・評価スクリプト
└── results/             # baseline vs PFP の比較結果（MAE/R²、プロット）
```

# Metal_complex（遷移金属錯体 / tmQM）

tmQMデータセット（遷移金属錯体、86,000錯体、TPSSh-D3BJ/def2-SVP、GFN2-xTB構造最適化）を対象に、
EGNN×PFPによる物性予測を行います（論文arXiv:2602.03046 Table 4に対応）。

## 物性一覧（5物性）

| フォルダ | 物性 | 説明 |
|---|---|---|
| [`tmQM_dipole`](./tmQM_dipole) | μ | 双極子モーメント（tmQM, 86,000錯体） |
| [`tmQM_HOMO`](./tmQM_HOMO) | εHOMO | HOMOエネルギー（tmQM, 86,000錯体） |
| [`tmQM_LUMO`](./tmQM_LUMO) | εLUMO | LUMOエネルギー（tmQM, 86,000錯体） |
| [`tmQM_HOMO_LUMO_gap`](./tmQM_HOMO_LUMO_gap) | Δε | HOMO-LUMOギャップ（tmQM, 86,000錯体） |
| [`tmQM_metal_charge`](./tmQM_metal_charge) | Metal_q | 金属上の部分電荷（tmQM, 86,000錯体） |

各フォルダ構成は以下の通りです。

```
tmQM_xxx/
├── README.md   # 物性の詳細・結果サマリ（出典: 論文Table 4）
├── scripts/    # グラフ構築・baseline学習・PFP学習・評価スクリプト
└── results/    # baseline vs PFP の比較結果プロット
```

元データ・PFP記述子・学習済みグラフ/チェックポイントはサイズの都合上、本リポジトリには含めずローカルでのみ管理しています。

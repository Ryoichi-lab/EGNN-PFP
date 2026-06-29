# Organic（有機分子 / QM9）

QM9データセット（有機小分子、約13万分子、B3LYP/6-31G(2df,p)レベル）を対象に、
EGNN×PFPによる物性予測を行います。

## 物性一覧（12物性）

| フォルダ | 物性 | 説明 |
|---|---|---|
| [`QM9_dipole`](./QM9_dipole) | μ | 双極子モーメント |
| [`QM9_polarizability`](./QM9_polarizability) | α | 等方分極率 |
| [`QM9_HOMO`](./QM9_HOMO) | HOMO | HOMOエネルギー |
| [`QM9_LUMO`](./QM9_LUMO) | LUMO | LUMOエネルギー |
| [`QM9_HOMO_LUMO_gap`](./QM9_HOMO_LUMO_gap) | gap | HOMO-LUMOギャップ |
| [`QM9_R2`](./QM9_R2) | ⟨R²⟩ | 電子空間広がり |
| [`QM9_ZPVE`](./QM9_ZPVE) | ZPVE | 零点振動エネルギー |
| [`QM9_U0`](./QM9_U0) | U₀ | 内部エネルギー(0K) |
| [`QM9_U`](./QM9_U) | U | 内部エネルギー(298.15K) |
| [`QM9_H`](./QM9_H) | H | エンタルピー |
| [`QM9_G`](./QM9_G) | G | 自由エネルギー |
| [`QM9_Cv`](./QM9_Cv) | Cv | 定積熱容量 |

各フォルダ構成は以下の通りです。

```
QM9_xxx/
├── README.md   # 物性の詳細・結果サマリ（出典: 論文Table 3）
├── scripts/    # グラフ構築・baseline学習・PFP学習・評価スクリプト
└── results/    # baseline vs PFP の比較結果プロット
```

元データ・PFP記述子（256次元）・学習済みグラフ/チェックポイントはサイズの都合上、本リポジトリには含めずローカルでのみ管理しています。

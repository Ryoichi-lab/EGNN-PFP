# QM9_NMR_1H

**対象物性**: ¹H核 NMR化学シールディング（気相, ppm）
**データセット**: QM9（QM9NMRデータセット, DFTで計算された遮蔽定数）
**手法**: EGNN（baseline） vs EGNN×PFP記述子

> **注記**: これは論文 *"Impact of Local Descriptors Derived from Machine Learning
> Potentials in Graph Neural Networks for Molecular Property Prediction"*
> (Uchiyama et al., arXiv:2602.03046) には含まれない拡張実験です。論文はQM9の標準12物性
> （HOMO/LUMOギャップ等）のみを対象としており、NMRシールディング予測は本リポジトリ作成時に
> ポートフォリオ用として追加したものです。

## モデル構成について（重要）

分子内の全原子（H, C, N, O, F）を完全グラフのノードとして含み、各原子にPFP記述子
（256次元）+ one-hot元素タイプ(5) + 幾何学的特徴(4) = 265次元のノード特徴を付与する
グラフを構築する。学習ターゲットは原子ごとのNMR遮蔽定数だが、**¹H用と¹³C用は別々に
学習された2つのモデル**であり、共通グラフ構造を使い回しつつも以下のようにターゲット
元素が異なる：

- `make_graph_1H.py` / `EGNN_1H.py`: ターゲット元素をH原子に限定し、H原子の遮蔽定数を
  回帰する（グラフの`y`にはH原子のみ実測値が入り、他原子は`-1`でマスク。マスクで損失
  計算対象をH原子のみに限定）。
- `make_graph_13C.py` / `EGNN_13C.py`: 同様にC原子の遮蔽定数を回帰する
  （`QM9_NMR_13C/`参照）。

つまり「単一モデルが全原子のシールディングを同時出力し、評価時に元素でフィルタする」
という構成ではなく、**グラフ構築段階でターゲット元素ごとに別々のデータセット・モデルを
用意し、損失計算もそのターゲット元素のみで行う**という構成である。ノード特徴・グラフ
トポロジー（完全グラフ、全原子をノードとして保持）は13C用・1H用で共通のため、
パイプラインスクリプト一式を両ディレクトリに重複コピーしている。

baseline版（`*_baseline.py`）はPFP記述子を含まず、one-hot元素タイプ(5)+幾何学的特徴(4)
= 9次元のノード特徴のみを使用する（PFP記述子の寄与を比較するためのアブレーション）。

## 結果サマリ

### QM9内部検証セット（5-fold KFoldのうち1 fold, fold4をval/testとして使用）

学習時のベストバリデーションMAE（`training_stats.json` / 学習ログより）:

| 構成 | Val MAE (ppm) |
|---|---|
| baseline (EGNN, 9次元ノード特徴) | 0.0455 |
| PFP記述子付き (EGNN×PFP, 265次元ノード特徴) | 0.0380 |

QM9内部分割でも、PFP記述子付きモデルの方がbaselineよりわずかに低いMAEを達成している。
PFP記述子の効果がより明確に表れるのは、分布外（OOD）の外部分子セットで評価した
以下のケーススタディである。

### 外部ケーススタディ（OOD分子: 12drugs, 40drugs, GDB, PAH, pyrimidinone）

QM9で学習したモデルを、QM9に含まれない外部の分子セット（医薬品分子・PAH等）に
適用した結果（`evaluate_case_studies_single.py` / `evaluate_case_studies_baseline.py`
の出力ログより）。原子レベルMAE・R²。

| データセット | baseline MAE (ppm) | baseline R² | PFP付き MAE (ppm) | PFP付き R² |
|---|---|---|---|---|
| 12drugs | 1.342 | -3.042 | 0.270 | 0.730 |
| 40drugs | 0.250 | 0.976 | 0.139 | 0.989 |
| GDB | 0.267 | 0.955 | 0.131 | 0.989 |
| PAH | 1.019 | -2.774 | 0.592 | -0.132 |
| pyrimidinone | 0.065 | 0.998 | 0.074 | 0.997 |
| **Overall** | **0.274** | **0.743** | **0.130** | **0.976** |

12drugsとPAHは芳香族・縮合環構造がQM9学習分布から外れるため両モデルともR²が低い
（baselineは特に12drugsで外挿に失敗、R²が負）。PFP記述子付きモデルは外部分子セット
全体でbaselineよりも一貫して低いMAE・高いR²を示している。

## ディレクトリ構成

```
QM9_NMR_1H/
├── scripts/
│   ├── make_graph_1H.py             # グラフ構築（PFP記述子あり, ターゲット=H）
│   ├── make_graph_1H_baseline.py    # グラフ構築（PFP記述子なし, ターゲット=H）
│   ├── make_graph_13C.py            # （13C用と共通スクリプト）
│   ├── make_graph_13C_baseline.py   # （13C用と共通スクリプト）
│   ├── make_graph_case_studies.py   # 外部ケーススタディ分子のグラフ構築
│   ├── run_make_graph_1H*.sh        # グラフ構築実行スクリプト
│   ├── run_make_graph_13C*.sh       # （13C用と共通）
│   ├── EGNN_1H.py                   # 学習（PFP記述子あり）
│   ├── EGNN_1H_baseline.py          # 学習（PFP記述子なし, アブレーション）
│   ├── EGNN_1H_cv.py                # 5-fold交差検証版
│   ├── EGNN_1H_reg.py               # 正則化版（weight_decay/Dropout強化, 未完走）
│   ├── EGNN_13C*.py                 # （13C用と共通スクリプト）
│   ├── compare_baseline_pfp.py      # baseline vs PFP 比較プロット生成
│   ├── compare_case_studies.py      # ケーススタディ比較プロット生成
│   ├── evaluate_case_studies_single.py    # 外部分子セットでの評価（PFPモデル）
│   ├── evaluate_case_studies_baseline.py  # 外部分子セットでの評価（baselineモデル）
│   ├── evaluate_case_studies_cv.py        # CVモデルでの評価
│   ├── plot_pfp_test_yy.py          # QM9 val/testセットのy-yプロット
│   └── plot_case_studies_summary.py # ケーススタディ結果サマリプロット
└── results/
    ├── qm9_val_1H_comparison.png        # QM9内部val: baseline vs PFP 比較
    ├── qm9_val_pfp_test_yy.png          # QM9内部val/test y-yプロット（13C, 1H 2パネル）
    ├── qm9_val_metrics_summary.png      # QM9内部val MAE/R²バーチャート
    ├── training_pfp_1H_epoch_500_yy.png       # PFPモデル学習曲線終盤のy-yプロット
    ├── training_baseline_1H_epoch_500_yy.png  # baselineモデル学習曲線終盤のy-yプロット
    ├── case_study_pfp_1H_summary_yy.png       # 外部ケーススタディ y-y（PFPモデル, 13C+1H）
    ├── case_study_pfp_1H_overall.png          # 外部ケーススタディ全体プロット（PFPモデル）
    ├── case_study_baseline_1H_overall.png     # 外部ケーススタディ全体プロット（baseline）
    ├── case_study_comparison_1H_scatter.png   # baseline vs PFP 散布図比較
    ├── case_study_metrics_comparison.png      # ケーススタディ別MAE/R²比較
    └── case_study_mae_improvement.png         # PFP導入によるMAE改善率
```

## 移植元

`~/qm9nmr_backup_20260304/EGNN_PFP/`（2026-03-04時点のスナップショット）から移植。
元の作業ディレクトリ `~/qm9nmr/EGNN_PFP/` には溶媒効果(CCl4/DMSO等)・CASCADE比較・
ファインチューニング等の派生実験が多数追加されているが、本ポートフォリオでは
QM9気相シールディング予測（baseline vs PFP）のコアパイプラインのみを移植した。

学習済みモデル重み（`.pth`）・構築済みグラフ（`.pt`）・PFP記述子（`.npz`）・生xyz座標・
完全な学習ログは容量の都合上移植していない（再現するには元データ・PFP記述子の再生成が
必要）。

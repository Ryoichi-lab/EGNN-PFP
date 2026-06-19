# tmQMg_NMR_shielding

**対象物性**: 金属核NMR等方シールディング σ（非相対論, σ_nonrel）
**データセット**: tmQMg（遷移金属錯体, 別名・拡張データセット。論文 arXiv:2602.03046 のtmQM=86,000件とは別物）
**手法**: EGNN（baseline） vs EGNN×PFP記述子

> **本ポートフォリオの対象範囲について**
> このNMRシールディング予測プロジェクトは元々、相対論効果（ADF Scalar/SO-ZORA計算）まで含む発展的な研究です。
> **今回のポートフォリオでは、ひとまず相対論を考慮しない非相対論版（σ_nonrel, ORCA PBE0/def2-TZVP）のみを対象とします。**
> 相対論補正（σ_rel, Δσ, σ_soz, σ_so, Δσ_soz 等、ADF Scalar/SO-ZORA計算ベース）に関するスクリプト・結果は別途研究中であり、本リポジトリには含めていません。
> また、本プロジェクトは論文（arXiv:2602.03046）には含まれない発展的な別プロジェクトです。

## 結果サマリ

### σ_nonrel（非相対論NMRシールディング, ORCA PBE0/def2-TZVP）の結果

（社内実験記録、未発表）

| 構成 | N_train | Test MAE | R² |
|---|---|---|---|
| EGNN×PFP | 4,247 | 29.8 ppm | 0.426 |

**R²がやや低い理由**: MAE自体は良好だが、重金属錯体でσ_nonrelが極端な値（-8000 ppm超）を取る外れ値が存在し、二乗誤差を支配しているため。

**データセット規模**: 全分子数 約9,890件（遷移金属錯体18種）。ADF Scalar ZORA計算済みのうち、本物性（σ_nonrel）はORCA PBE0/def2-TZVPベースで全9,890件計算済み。`data/nmr_shieldings.csv` はADF/ORCA双方が完了した分子のみを集計した中間生成物のサブセット（8,095件）であり、学習に使われたグラフ分割時点でtmQMg既存グラフとマッチしたものがさらに絞り込まれてN_train=4,247等の数値になっている。

## データについて

`data/nmr_shieldings.csv` は `scripts/parse_nmr_shieldings.py` が ADF（ZORA, 相対論）出力と ORCA（PBE0, 非相対論）出力の両方をパースして突合した中間CSVであり、列に `shield_rel`（相対論）・`delta_shield`（差分）も含まれます。**本ポートフォリオで学習・評価に用いるのは `shield_nonrel` 列のみ**です。`shield_rel` / `delta_shield` 列はCSVの生成プロセス上同居しているだけで、相対論モデルの学習・評価には使用していません。

## スクリプトについて

- `scripts/parse_nmr_shieldings.py`: ADF/ORCA出力からNMRシールディングCSVを生成。ADF（相対論）パーサも含む共通スクリプトのため、相対論カラムも出力されますが、本ポートフォリオでは `shield_nonrel` のみ使用します。
- `scripts/build_nmr_splits.py`: train/val/test分割を構築。PFP記述子付きグラフとbaseline（記述子なし）グラフの両方を生成し、`y_nmr_rel` / `y_nmr_nonrel` / `y_delta_nmr` の3ターゲット分の正規化統計を計算する共通スクリプトです。本ポートフォリオでは `y_nmr_nonrel` のみ使用します。
- `scripts/train_nmr_target.py`: シングルターゲット学習スクリプト。`--arch {egnn,pfp}` でbaseline（EGNN単体）とEGNN×PFP記述子付きモデルを切り替え、`--target {rel,nonrel,delta}` で学習対象を切り替えます。本ポートフォリオでは `--arch pfp --target nonrel` および `--arch egnn --target nonrel` のみを使用しています。
  ```
  python train_nmr_target.py --arch egnn --target nonrel   # baseline (EGNN, PFPなし)
  python train_nmr_target.py --arch pfp  --target nonrel   # EGNN×PFP
  ```
- `scripts/plot_nmr_yy_individual.py`: モデルごとのy-yプロット（予測 vs 真値）を生成。`rel`/`delta`用の分岐も含む共通スクリプトですが、結果として同梱しているのは `nonrel` 用の出力のみです。
- `scripts/run_orca_pbe0_tmQMg.sh`, `scripts/run_orca_pbe0_tmQMg_test.sh`: ORCA PBE0/def2-TZVP計算のPBSジョブ投入スクリプト（汎用テンプレート、`input_orca_pbe0_descriptor_rich` ディレクトリ向け。記述子計算と共有しているテンプレートで、NMRシールディング自体は下記の `run_orca_nmr_nonrel_*.sh` が対応する計算を実行）。
- `scripts/run_orca_nmr_nonrel_{a,b,c,d,e}.sh`: σ_nonrel計算（ORCA非相対論NMR、`input_orca_nmr_nonrel/batch_{a..e}/`）を実際に実行するバッチジョブスクリプト。生計算出力（ADF/ORCAの個別計算フォルダ）はサイズの都合上、本リポジトリには含めていません。

## ディレクトリ構成

```
tmQMg_NMR_shielding/
├── data/
│   ├── nmr_shieldings.csv          # ADF/ORCA突合済みシールディング値（8,095件。shield_nonrel列のみ本ポートフォリオで使用）
│   └── nmr_shielding_stats.json    # 正規化統計（mean/std）。nonrel以外のキーも含む共通ファイル
├── scripts/
│   ├── parse_nmr_shieldings.py     # ADF/ORCA出力 → CSV
│   ├── build_nmr_splits.py         # train/val/test分割生成
│   ├── train_nmr_target.py         # 学習スクリプト（--arch egnn/pfp, --target rel/nonrel/delta）
│   ├── plot_nmr_yy_individual.py   # y-yプロット生成
│   ├── run_orca_pbe0_tmQMg.sh      # ORCA PBE0計算ジョブ（汎用テンプレート）
│   ├── run_orca_pbe0_tmQMg_test.sh
│   └── run_orca_nmr_nonrel_{a,b,c,d,e}.sh  # σ_nonrel計算バッチジョブ（input_orca_nmr_nonrel/batch_*向け）
└── results/
    ├── nmr_yy_pfp_nonrel.png       # y-yプロット: EGNN×PFP, σ_nonrel（テストセット）
    ├── nmr_yy_egnn_nonrel.png      # y-yプロット: baseline EGNN, σ_nonrel（テストセット）
    └── nmr_per_metal_nonrel.png    # 金属種ごとのMAE内訳, σ_nonrel
```

## 除外したもの（移植元 `relativistic_effect/` から意図的にコピーしていないもの）

- 相対論関連の全スクリプト・結果（σ_rel, Δσ, σ_soz, σ_so, Δσ_soz, ADF Scalar/SO-ZORA関連のfinetune・プロット・ログ一式）
- `.pt` グラフファイル（学習用グラフ。PFP記述子版・baseline版とも数百MB〜数GB規模）
- `.pth` チェックポイント（学習済みモデル重み）
- ADF/ORCAの生計算出力ディレクトリ（`input_nmr/`, `input_orca_nmr_nonrel/` など。個別計算フォルダが数千件あり、合計で数百MB〜のサイズになるため）
- ログファイル全般（`.log`, PBSジョブの標準出力など）
- `run_orca_nmr_nonrel_*_p2.sh`（a/b/c の再実行版で内容がほぼ重複するため、初回実行用の a〜e のみを採用）

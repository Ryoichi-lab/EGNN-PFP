# EGNN-PFP

PFP（Matlantis社の汎用ニューラルポテンシャル"Preferred Potential"）が学習した原子embedding（記述子）を、
E(n)-Equivariant Graph Neural Network（EGNN）の入力特徴として活用し、量子化学計算で得られる各種物性値を
高速かつ高精度に予測するプロジェクトです。

## モチベーション

DFT等の量子化学計算は高精度ですが計算コストが大きく、大規模なスクリーニングには不向きです。
本プロジェクトでは、汎用ニューラルポテンシャルPFPが内部に持つ原子環境の情報（256次元の記述子）を
追加の条件付け特徴としてEGNNに組み込むことで、ベースライン（PFP記述子なし）のEGNNと比較して
予測精度がどの程度向上するかを検証しています。

## 構成

| ディレクトリ | 対象 | データセット |
|---|---|---|
| [`Organic/`](./Organic) | 有機分子 | QM9 |
| [`Metal_complex/`](./Metal_complex) | 遷移金属錯体 | tmQMg |

各物性フォルダには、データ前処理・グラフ構築・学習・評価のスクリプトと結果（ベースライン vs PFP記述子付与の比較）を格納しています。

## 共通手法

- アーキテクチャ: EGNN（E(n)-Equivariant Graph Neural Network）
- 入力特徴: 原子番号・座標に加え、PFP記述子（256次元）をノード特徴として付与
- 比較: PFP記述子なし（baseline）と PFP記述子あり（pfp）の2構成でMAE/R²を比較

# Paper

本リポジトリの手法・実験は以下の論文に基づきます。

**Impact of Local Descriptors Derived from Machine Learning Potentials in Graph Neural Networks for Molecular Property Prediction**

Ryoichi Uchiyama, Yuya Nakajima, Yuta Tanaka, Junji Seino

- Department of Chemistry and Biochemistry, School of Advanced Science and Engineering, Waseda University
- Waseda Research Institute for Science and Engineering, Waseda University
- AI Innovation Department, ENEOS Holdings, Inc.

PFP（Matlantisで取得したPreferred Potentialの原子embedding特徴量）をEGNN(E(n)-Equivariant Graph Neural Network)のノード特徴に組み込む"EGNN-PFP"モデルを提案し、QM9（有機分子12物性）およびtmQM（遷移金属錯体5物性）で、PFP記述子なしのEGNN baselineおよび既存3D GNN手法（SchNet, DimeNet++, PaiNN, ViSNetなど）と比較評価。

- QM9: 12物性中11物性でPFP記述子ありが上回る（例: 双極子モーメントMAE 0.029→0.022 D、HOMO-LUMOギャップ47.6→40.4 meV）
- tmQM: 5物性(μ, Δε, ε_HOMO, ε_LUMO, Metal_q)すべてでPFP記述子ありが大幅に上回る（例: Metal_q MAE 0.0562→0.0196）

詳細な数値は本リポジトリ各物性フォルダのREADMEに転記しています（出典: Table 3, Table 4）。

**arXiv**: [arXiv:2602.03046](https://arxiv.org/abs/2602.03046)

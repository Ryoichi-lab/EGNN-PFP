#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QM9データセットから分極率 (α, a.u.) と
最適化構造座標 (Å) を抽出し、各分子を .xyz ファイルとして保存するスクリプト。
出典: B3LYP/6-31G(2df,p) (Ramakrishnan et al., Sci. Data, 2014)
"""

import os
from torch_geometric.datasets import QM9

# ==============================================================  
# 1. データセットをロード
# ==============================================================  

print("🔹 Loading QM9 dataset...")
dataset = QM9(root='data/QM9')
print(f"✓ Loaded {len(dataset)} molecules.")

# ==============================================================  
# 2. 出力ディレクトリの設定
# ==============================================================  

output_dir = "/home/users/uchiyama/QM9_polarizability/qm9_xyz_polarizability"
os.makedirs(output_dir, exist_ok=True)

# ==============================================================  
# 3. 物性インデックス（分極率 'alpha'）を特定
# ==============================================================  

# torch_geometric.datasets.QM9 では 19種類のラベルを持つ
# 標準的には以下の順序:
# ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', ...]

try:
    alpha_index = dataset.target.index('alpha')
except AttributeError:
    # torch_geometricのQM9では target 属性が存在しないため、index=1が分極率
    alpha_index = 1

print(f"🔹 Polarizability index = {alpha_index}")

# ==============================================================  
# 4. 元素番号 → 元素記号対応表
# ==============================================================  

periodic_table = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca"
]

# ==============================================================  
# 5. 各分子を .xyz ファイルとして出力
# ==============================================================  

for i, data in enumerate(dataset):
    # --- 分極率を安全に取得（19要素対応） ---
    if data.y.ndim == 2:
        alpha = float(data.y[0, alpha_index])
    else:
        alpha = float(data.y[alpha_index])

    # --- 原子情報 ---
    Z = data.z.tolist()       # 原子番号
    pos = data.pos.tolist()   # 座標 (Å)
    symbols = [periodic_table[z - 1] for z in Z]

    # --- ファイル名と書き出し ---
    filename = os.path.join(output_dir, f"mol_{i:06d}.xyz")
    with open(filename, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"polarizability={alpha:.6f} a.u.\n")
        for s, (x, y, zc) in zip(symbols, pos):
            f.write(f"{s:2s}  {x: .6f}  {y: .6f}  {zc: .6f}\n")

    if (i + 1) % 5000 == 0:
        print(f"✓ {i + 1} molecules written...")

print(f"✅ 完了: {len(dataset)}個の分子を '{output_dir}' に保存しました。")

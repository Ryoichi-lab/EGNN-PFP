#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QM9データセットからHOMO-LUMOギャップ (gap, eV) と
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

output_dir = "/home/users/uchiyama/QM9_HOMO_LUMO/qm9_xyz_gap"
os.makedirs(output_dir, exist_ok=True)

# ==============================================================  
# 3. 物性インデックス（HOMO-LUMOギャップ 'gap'）を特定
# ==============================================================  

# torch_geometric.datasets.QM9 のラベル順序:
# ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', ...]

try:
    gap_index = dataset.target.index('gap')
except AttributeError:
    # PyTorch GeometricのQM9では固定順序: index=4 が gap
    gap_index = 4

print(f"🔹 HOMO-LUMO gap index = {gap_index}")

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
    # --- gapを安全に取得（19要素対応） ---
    if data.y.ndim == 2:
        gap = float(data.y[0, gap_index])
    else:
        gap = float(data.y[gap_index])

    # --- 原子情報 ---
    Z = data.z.tolist()       # 原子番号
    pos = data.pos.tolist()   # 座標 (Å)
    symbols = [periodic_table[z - 1] for z in Z]

    # --- ファイル名と書き出し ---
    filename = os.path.join(output_dir, f"mol_{i:06d}.xyz")
    with open(filename, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"gap={gap:.6f} eV\n")
        for s, (x, y, zc) in zip(symbols, pos):
            f.write(f"{s:2s}  {x: .6f}  {y: .6f}  {zc: .6f}\n")

    if (i + 1) % 5000 == 0:
        print(f"✓ {i + 1} molecules written...")

print(f"✅ 完了: {len(dataset)}個の分子を '{output_dir}' に保存しました。")

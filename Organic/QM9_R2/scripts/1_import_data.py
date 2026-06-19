#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QM9データセットから電子密度二乗平均回転半径 (R², bohr²) と
最適化構造座標 (Å) を抽出し、
各分子を .xyz ファイルとして保存するスクリプト。

出典: B3LYP/6-31G(2df,p) (Ramakrishnan et al., Sci. Data, 2014)
注意: R²は原子化エネルギー補正を適用しない（EGNN論文準拠）
"""

import os
from torch_geometric.datasets import QM9

print("="*60)
print("QM9 R² データ抽出（補正なし）")
print("="*60)

# ==============================================================
# 1. データセットをロード
# ==============================================================

print("\n🔹 Loading QM9 dataset...")
dataset = QM9(root='data/QM9')
print(f"✓ Loaded {len(dataset)} molecules.")

# ==============================================================
# 2. 出力ディレクトリの設定
# ==============================================================

output_dir = "/home/users/uchiyama/QM9_R2/qm9_xyz_R2"
os.makedirs(output_dir, exist_ok=True)

# ==============================================================
# 3. 物性インデックス（R²）を特定
# ==============================================================

# PyG QM9 のラベル順（5番目が R²）
# ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
#  'zpve', 'U0', 'U', 'H', 'G', 'Cv']

R2_index = 5
print(f"🔹 R² index = {R2_index}")
print(f"🔹 原子化エネルギー補正: なし（構造依存量のため）")
print(f"   R²は電子密度の空間的広がりを表し、原子数に単純比例しない")

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

R2_stats = {
    'total_molecules': 0,
    'min_R2': float('inf'),
    'max_R2': float('-inf'),
    'sum_R2': 0.0
}

for i, data in enumerate(dataset):

    # --- R² を取得（19要素の data.y に対応） ---
    if data.y.ndim == 2:
        R2_value = float(data.y[0, R2_index])
    else:
        R2_value = float(data.y[R2_index])

    # --- 原子情報 ---
    Z = data.z.tolist()
    pos = data.pos.tolist()
    symbols = [periodic_table[z - 1] for z in Z]

    # 統計情報を記録
    R2_stats['total_molecules'] += 1
    R2_stats['min_R2'] = min(R2_stats['min_R2'], R2_value)
    R2_stats['max_R2'] = max(R2_stats['max_R2'], R2_value)
    R2_stats['sum_R2'] += R2_value

    # --- ファイル名 ---
    filename = os.path.join(output_dir, f"mol_{i:06d}.xyz")

    # --- 書き出し ---
    with open(filename, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"R2={R2_value:.6f} bohr^2\n")  # ★ 補正なしの生データ
        for s, (x, y, zc) in zip(symbols, pos):
            f.write(f"{s:2s}  {x: .6f}  {y: .6f}  {zc: .6f}\n")

    if (i + 1) % 5000 == 0:
        print(f"✓ {i + 1} molecules written...")

# ==============================================================
# 6. 統計を表示
# ==============================================================

print(f"\n✅ 完了: {len(dataset)}個の分子を '{output_dir}' に保存しました。")
print(f"\n📊 R² 統計:")
print(f"  総分子数: {R2_stats['total_molecules']}")
print(f"  平均R²: {R2_stats['sum_R2']/R2_stats['total_molecules']:.4f} bohr²")
print(f"  R²範囲: {R2_stats['min_R2']:.4f} ~ {R2_stats['max_R2']:.4f} bohr²")
print(f"\n✅ 補正なし（原データそのまま - EGNN/SchNet論文準拠）")
print(f"   理由: R²は構造依存量で原子数に単純比例しない")
print("="*60)
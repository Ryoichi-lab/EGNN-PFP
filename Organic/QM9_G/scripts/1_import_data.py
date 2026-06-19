#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
【修正完了版】QM9 自由エネルギー G (298.15K) 変換スクリプト
正しい原子参照エネルギー(G)を使用し、eVからHartreeへ変換して補正
"""

import os
import logging
from torch_geometric.datasets import QM9
from tqdm import tqdm

# ==============================================================
# 1. 設定
# ==============================================================

# 保存先ディレクトリ
OUTPUT_DIR = "/home/users/uchiyama/QM9_G/qm9_xyz_g"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 原子参照エネルギー (単位: Hartree)
# ユーザー様提供のログデータより
ATOM_REF_G_HA = {
    'H': -0.510927,
    'C': -37.861317,
    'N': -54.598897,
    'O': -75.079532,
    'F': -99.733544
}

# 単位変換係数 (eV -> Hartree)
# PyGはeVで読み込むため、これを掛けてHartreeに戻す
EV_TO_HARTREE = 1.0 / 27.211386246

ATOMIC_NUM_TO_SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}

# QM9 datasetのインデックス: G (Free Energy at 298.15K) は 10番目
# (0:mu, 1:alpha, 2:homo, 3:lumo, 4:gap, 5:r2, 6:zpve, 7:U0, 8:U, 9:H, 10:G)
G_INDEX = 10

# ==============================================================
# 2. データセットロード
# ==============================================================

print("🔹 Loading QM9 dataset (PyG default unit: eV)...")
dataset = QM9(root='data/QM9')
print(f"✓ Loaded {len(dataset)} molecules.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================
# 3. 変換と保存
# ==============================================================

print(f"\n🔹 原子化エネルギー(G)への変換(eV -> Ha -> Atomization)を開始...")
print(f"   Target Property: G (Free Energy at 298.15K)")
print(f"   Ref Check (C): {ATOM_REF_G_HA['C']} Ha")

count = 0
skipped = 0
debug_printed = False

for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
    # --- 1. PyGの生データ (単位: eV) ---
    g_ev = data.y[0, G_INDEX].item()
    
    # --- 2. Hartreeに変換 ---
    g_raw_ha = g_ev * EV_TO_HARTREE
    
    # --- 3. 原子情報の取得 ---
    atomic_numbers = data.z.numpy()
    positions = data.pos.numpy()
    
    # --- 4. 原子参照値の合計 (単位: Hartree) ---
    g_ref_sum_ha = 0.0
    atoms_data = []
    has_unknown = False
    
    for i, z in enumerate(atomic_numbers):
        element = ATOMIC_NUM_TO_SYMBOL.get(int(z), 'X')
        
        if element in ATOM_REF_G_HA:
            g_ref_sum_ha += ATOM_REF_G_HA[element]
        else:
            has_unknown = True
            break
            
        x, y, z_coord = positions[i]
        atoms_data.append((element, x, y, z_coord))
    
    if has_unknown:
        skipped += 1
        continue
    
    # --- 5. 補正計算 (Atomization Energy in Hartree) ---
    # G_corr = G_molecule(Ha) - Σ G_atom(Ha)
    g_corrected_ha = g_raw_ha - g_ref_sum_ha
    
    # デバッグ表示（最初の1分子だけ）
    if not debug_printed:
        print("\n🔍 --- 単位変換・補正チェック (Mol ID: 0) ---")
        print(f"   Raw G (PyG):          {g_ev:.4f} eV")
        print(f"   Raw G (Conv):         {g_raw_ha:.4f} Ha")
        print(f"   Ref Sum (Atoms):      {g_ref_sum_ha:.4f} Ha")
        print(f"   Corrected (Atomiz.):  {g_corrected_ha:.6f} Ha")
        
        # チェック: 原子化エネルギーは通常マイナスで、絶対値が大きすぎないか(10000とかになっていないか)
        if abs(g_corrected_ha) > 20.0:
             print("❌ エラー: 値が大きすぎます！単位変換がうまくいっていません。")
        else:
             print("✅ 正常: 値が原子化エネルギーの範囲(数Ha)に収まりました。")
        print("----------------------------------\n")
        debug_printed = True
    
    # --- 6. XYZ保存 ---
    out_filename = f"mol_{idx:06d}.xyz"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    
    num_atoms = len(atoms_data)
    
    with open(out_path, "w") as out_f:
        out_f.write(f"{num_atoms}\n")
        # 小文字 g で保存
        out_f.write(f"g={g_corrected_ha:.8f} Ha\n")
        
        for atom in atoms_data:
            out_f.write(f"{atom[0]:<2}  {atom[1]: .8f}  {atom[2]: .8f}  {atom[3]: .8f}\n")
    
    count += 1

print("\n" + "="*50)
print(f"✅ 完了: {count}個の分子を保存しました。")
print(f"スキップ: {skipped}個")
print(f"保存先: {OUTPUT_DIR}")
print("="*50)
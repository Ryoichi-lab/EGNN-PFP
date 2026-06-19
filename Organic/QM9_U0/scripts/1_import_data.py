#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
【修正完了版】QM9 原子化エネルギー変換スクリプト
PyGの仕様(eV)に合わせて、Hartreeに変換してから補正を行うバージョン
"""

import os
import logging
from torch_geometric.datasets import QM9
from tqdm import tqdm

# ==============================================================
# 1. 設定
# ==============================================================

OUTPUT_DIR = "/home/users/uchiyama/QM9_U0/qm9_xyz_u0"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 原子参照エネルギー (単位: Hartree) - B3LYP/6-31G(2df,p)
ATOM_REF_U0_HA = {
    'H': -0.500273,
    'C': -37.846772,
    'N': -54.583861,
    'O': -75.064579,
    'F': -99.718730
}

# 単位変換係数
EV_TO_HARTREE = 1.0 / 27.211386246

ATOMIC_NUM_TO_SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
U0_INDEX = 7 

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

print(f"\n🔹 原子化エネルギーへの変換(eV -> Ha -> Atomization)を開始...")

count = 0
skipped = 0
debug_printed = False

for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
    # --- 1. PyGの生データ (単位: eV) ---
    u0_ev = data.y[0, U0_INDEX].item()
    
    # --- 2. Hartreeに変換 ---
    u0_raw_ha = u0_ev * EV_TO_HARTREE
    
    # --- 3. 原子情報の取得 ---
    atomic_numbers = data.z.numpy()
    positions = data.pos.numpy()
    
    # --- 4. 原子参照値の合計 (単位: Hartree) ---
    u0_ref_sum_ha = 0.0
    atoms_data = []
    has_unknown = False
    
    for i, z in enumerate(atomic_numbers):
        element = ATOMIC_NUM_TO_SYMBOL.get(int(z), 'X')
        
        if element in ATOM_REF_U0_HA:
            u0_ref_sum_ha += ATOM_REF_U0_HA[element]
        else:
            has_unknown = True
            break
            
        x, y, z_coord = positions[i]
        atoms_data.append((element, x, y, z_coord))
    
    if has_unknown:
        skipped += 1
        continue
    
    # --- 5. 補正計算 (Atomization Energy in Hartree) ---
    # 両方とも Hartree 単位になったので引き算してOK
    u0_corrected_ha = u0_raw_ha - u0_ref_sum_ha
    
    # デバッグ表示（最初の1分子だけ）
    if not debug_printed:
        print("\n🔍 --- 単位変換・補正チェック (Mol ID: 0) ---")
        print(f"   Raw U0 (PyG):         {u0_ev:.4f} eV")
        print(f"   Raw U0 (Conv):        {u0_raw_ha:.4f} Ha")
        print(f"   Ref Sum (Atoms):      {u0_ref_sum_ha:.4f} Ha")
        print(f"   Corrected (Atomiz.):  {u0_corrected_ha:.6f} Ha")
        
        if abs(u0_corrected_ha) > 10.0:
            print("❌ エラー: 値がまだ大きすぎます！")
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
        out_f.write(f"u0={u0_corrected_ha:.8f} Ha\n")
        
        for atom in atoms_data:
            out_f.write(f"{atom[0]:<2}  {atom[1]: .8f}  {atom[2]: .8f}  {atom[3]: .8f}\n")
    
    count += 1

print("\n" + "="*50)
print(f"✅ 完了: {count}個の分子を保存しました。")
print(f"スキップ: {skipped}個")
print(f"保存先: {OUTPUT_DIR}")
print("="*50)
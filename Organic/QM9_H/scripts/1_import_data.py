#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
【決定版】QM9 エンタルピー H (Enthalpy at 298.15K) 変換スクリプト
あなたの環境で確認された atomref 値を使用
"""

import os
import logging
from torch_geometric.datasets import QM9
from tqdm import tqdm

# ==============================================================
# 1. 設定
# ==============================================================

OUTPUT_DIR = "/home/users/uchiyama/QM9_H/qm9_xyz_h"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 原子参照エネルギー (単位: Hartree) - Enthalpy H at 298.15K
# ログに出力された正確な値を使用
ATOM_REF_H_HA = {
    'H': -0.497912,
    'C': -37.844411,
    'N': -54.581501,
    'O': -75.062219,
    'F': -99.716370  # ログの値を反映
}

# 単位変換係数 (eV -> Hartree)
EV_TO_HARTREE = 1.0 / 27.211386246

ATOMIC_NUM_TO_SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}

# QM9 datasetのインデックス: H (Enthalpy) は 9番目
H_INDEX = 9

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

print(f"\n🔹 原子化エンタルピー(H)への変換(eV -> Ha -> Atomization)を開始...")
print(f"   Target Property: H (Enthalpy at 298.15K)")
print(f"   Ref Check (H atom): {ATOM_REF_H_HA['H']} Ha")

count = 0
skipped = 0
debug_printed = False

for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
    # --- 1. PyGの生データ (単位: eV) ---
    h_ev = data.y[0, H_INDEX].item()
    
    # --- 2. Hartreeに変換 ---
    h_raw_ha = h_ev * EV_TO_HARTREE
    
    # --- 3. 原子情報の取得 ---
    atomic_numbers = data.z.numpy()
    positions = data.pos.numpy()
    
    # --- 4. 原子参照値の合計 (単位: Hartree) ---
    h_ref_sum_ha = 0.0
    atoms_data = []
    has_unknown = False
    
    for i, z in enumerate(atomic_numbers):
        element = ATOMIC_NUM_TO_SYMBOL.get(int(z), 'X')
        
        if element in ATOM_REF_H_HA:
            h_ref_sum_ha += ATOM_REF_H_HA[element]
        else:
            has_unknown = True
            break
            
        x, y, z_coord = positions[i]
        atoms_data.append((element, x, y, z_coord))
    
    if has_unknown:
        skipped += 1
        continue
    
    # --- 5. 補正計算 (Atomization Enthalpy) ---
    # H_atomization = H_total - Sum(H_atom_refs)
    h_corrected_ha = h_raw_ha - h_ref_sum_ha
    
    # デバッグ表示（最初の1分子だけ）
    if not debug_printed:
        print("\n🔍 --- 単位変換・補正チェック (Mol ID: 0) ---")
        print(f"   Raw H (PyG):          {h_ev:.4f} eV")
        print(f"   Raw H (Conv):         {h_raw_ha:.4f} Ha")
        print(f"   Ref Sum (Atoms):      {h_ref_sum_ha:.4f} Ha")
        print(f"   Corrected (Atomiz.):  {h_corrected_ha:.6f} Ha")
        
        # チェック: メタン(CH4)の場合、大体 -0.64 Ha 前後になるはずです
        if abs(h_corrected_ha) > 20.0:
            print("❌ エラー: 値が大きすぎます！")
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
        # h=... として書き込み
        out_f.write(f"h={h_corrected_ha:.8f} Ha\n")
        
        for atom in atoms_data:
            out_f.write(f"{atom[0]:<2}  {atom[1]: .8f}  {atom[2]: .8f}  {atom[3]: .8f}\n")
    
    count += 1

print("\n" + "="*50)
print(f"✅ 完了: {count}個の分子を保存しました。")
print(f"保存先: {OUTPUT_DIR}")
print("="*50)
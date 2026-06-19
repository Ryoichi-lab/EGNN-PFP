#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
【修正完了版】QM9 内部エネルギー U (298.15K) 変換スクリプト
正しい原子参照エネルギー(U)を使用
"""

import os
import logging
from torch_geometric.datasets import QM9
from tqdm import tqdm

# ==============================================================
# 1. 設定
# ==============================================================

OUTPUT_DIR = "/home/users/uchiyama/QM9_U/qm9_xyz_u"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 原子参照エネルギー (単位: Hartree) - B3LYP/6-31G(2df,p)
# atomref.txt の 3列目 (U at 298.15K) の値
# あなたのログに出ていた値が正解です
ATOM_REF_U_HA = {
    'H': -0.498857,
    'C': -37.845355,
    'N': -54.582445,
    'O': -75.063163,
    'F': -99.717314
}

# 単位変換係数 (eV -> Hartree)
EV_TO_HARTREE = 1.0 / 27.211386246

ATOMIC_NUM_TO_SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}

# QM9 datasetのインデックス: U (Internal Energy at 298.15K) は 8番目
U_INDEX = 8 

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

print(f"\n🔹 原子化エネルギー(U)への変換(eV -> Ha -> Atomization)を開始...")
print(f"   Target Property: U (Internal Energy at 298.15K)")
print(f"   Ref Check (H): {ATOM_REF_U_HA['H']} Ha")

count = 0
skipped = 0
debug_printed = False

for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
    # --- 1. PyGの生データ (単位: eV) ---
    u_ev = data.y[0, U_INDEX].item()
    
    # --- 2. Hartreeに変換 ---
    u_raw_ha = u_ev * EV_TO_HARTREE
    
    # --- 3. 原子情報の取得 ---
    atomic_numbers = data.z.numpy()
    positions = data.pos.numpy()
    
    # --- 4. 原子参照値の合計 (単位: Hartree) ---
    u_ref_sum_ha = 0.0
    atoms_data = []
    has_unknown = False
    
    for i, z in enumerate(atomic_numbers):
        element = ATOMIC_NUM_TO_SYMBOL.get(int(z), 'X')
        
        if element in ATOM_REF_U_HA:
            u_ref_sum_ha += ATOM_REF_U_HA[element]
        else:
            has_unknown = True
            break
            
        x, y, z_coord = positions[i]
        atoms_data.append((element, x, y, z_coord))
    
    if has_unknown:
        skipped += 1
        continue
    
    # --- 5. 補正計算 (Atomization Energy in Hartree) ---
    u_corrected_ha = u_raw_ha - u_ref_sum_ha
    
    # デバッグ表示（最初の1分子だけ）
    if not debug_printed:
        print("\n🔍 --- 単位変換・補正チェック (Mol ID: 0) ---")
        print(f"   Raw U (PyG):          {u_ev:.4f} eV")
        print(f"   Raw U (Conv):         {u_raw_ha:.4f} Ha")
        print(f"   Ref Sum (Atoms):      {u_ref_sum_ha:.4f} Ha")
        print(f"   Corrected (Atomiz.):  {u_corrected_ha:.6f} Ha")
        
        # チェック: 原子化エネルギーは通常 -0.5 〜 -10.0 くらいの範囲
        if abs(u_corrected_ha) > 20.0:
            print("❌ エラー: 値が大きすぎます！単位変換か参照値を確認してください。")
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
        out_f.write(f"u={u_corrected_ha:.8f} Ha\n")
        
        for atom in atoms_data:
            out_f.write(f"{atom[0]:<2}  {atom[1]: .8f}  {atom[2]: .8f}  {atom[3]: .8f}\n")
    
    count += 1

print("\n" + "="*50)
print(f"✅ 完了: {count}個の分子を保存しました。")
print(f"スキップ: {skipped}個")
print(f"保存先: {OUTPUT_DIR}")
print("="*50)
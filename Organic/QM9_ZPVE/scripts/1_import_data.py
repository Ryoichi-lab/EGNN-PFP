#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
【修正完了版】QM9 ZPVE (Zero Point Vibration Energy) 変換スクリプト
PyGの仕様(eV)に合わせて、Hartreeに変換するバージョン
※ ZPVEの原子参照値は0なので、実質的に単位変換のみを行います。
"""

import os
import logging
from torch_geometric.datasets import QM9
from tqdm import tqdm

# ==============================================================
# 1. 設定
# ==============================================================

OUTPUT_DIR = "/home/users/uchiyama/QM9_ZPVE/qm9_xyz_zpve"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ZPVE (零点振動エネルギー) の原子参照値
# 単原子は振動モードを持たないため、理論的にすべて 0 Hartree です。
ATOM_REF_ZPVE_HA = {
    'H': 0.0,
    'C': 0.0,
    'N': 0.0,
    'O': 0.0,
    'F': 0.0
}

# 単位変換係数 (eV -> Hartree)
EV_TO_HARTREE = 1.0 / 27.211386246

ATOMIC_NUM_TO_SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}

# QM9 datasetのインデックス: ZPVE は 6番目
ZPVE_INDEX = 6

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

print(f"\n🔹 ZPVEの変換(eV -> Ha)を開始...")
print(f"   Target Property: ZPVE")
print(f"   Ref Check (All atoms): 0.0 Ha")

count = 0
skipped = 0
debug_printed = False

for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
    # --- 1. PyGの生データ (単位: eV) ---
    zpve_ev = data.y[0, ZPVE_INDEX].item()
    
    # --- 2. Hartreeに変換 ---
    zpve_raw_ha = zpve_ev * EV_TO_HARTREE
    
    # --- 3. 原子情報の取得 ---
    atomic_numbers = data.z.numpy()
    positions = data.pos.numpy()
    
    # --- 4. 原子参照値の合計 (単位: Hartree) ---
    # ZPVEの場合は常に 0.0 になりますが、形式的に計算します
    zpve_ref_sum_ha = 0.0
    atoms_data = []
    has_unknown = False
    
    for i, z in enumerate(atomic_numbers):
        element = ATOMIC_NUM_TO_SYMBOL.get(int(z), 'X')
        
        if element in ATOM_REF_ZPVE_HA:
            zpve_ref_sum_ha += ATOM_REF_ZPVE_HA[element]
        else:
            has_unknown = True
            break
            
        x, y, z_coord = positions[i]
        atoms_data.append((element, x, y, z_coord))
    
    if has_unknown:
        skipped += 1
        continue
    
    # --- 5. 補正計算 (Atomization Energy in Hartree) ---
    # ZPVE_corr = ZPVE_total - 0.0
    zpve_corrected_ha = zpve_raw_ha - zpve_ref_sum_ha
    
    # デバッグ表示（最初の1分子だけ）
    if not debug_printed:
        print("\n🔍 --- 単位変換チェック (Mol ID: 0) ---")
        print(f"   Raw ZPVE (PyG):       {zpve_ev:.4f} eV")
        print(f"   Raw ZPVE (Conv):      {zpve_raw_ha:.6f} Ha")
        print(f"   Ref Sum (Atoms):      {zpve_ref_sum_ha:.6f} Ha (Should be 0.0)")
        print(f"   Final ZPVE:           {zpve_corrected_ha:.6f} Ha")
        
        # チェック: 平均的な分子のZPVEは 0.1 〜 0.2 Ha 程度
        if zpve_corrected_ha > 1.0:
            print("❌ エラー: 値が大きすぎます！単位変換を確認してください。")
        else:
            print("✅ 正常: 値が適切な範囲(0.1〜0.2 Ha程度)に収まりました。")
        print("----------------------------------\n")
        debug_printed = True
    
    # --- 6. XYZ保存 ---
    out_filename = f"mol_{idx:06d}.xyz"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    
    num_atoms = len(atoms_data)
    
    with open(out_path, "w") as out_f:
        out_f.write(f"{num_atoms}\n")
        # zpve=... として書き込み
        out_f.write(f"zpve={zpve_corrected_ha:.8f} Ha\n")
        
        for atom in atoms_data:
            out_f.write(f"{atom[0]:<2}  {atom[1]: .8f}  {atom[2]: .8f}  {atom[3]: .8f}\n")
    
    count += 1

print("\n" + "="*50)
print(f"✅ 完了: {count}個の分子を保存しました。")
print(f"スキップ: {skipped}個")
print(f"保存先: {OUTPUT_DIR}")
print("="*50)
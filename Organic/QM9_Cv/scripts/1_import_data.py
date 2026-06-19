#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch GeometricのQM9データセットから定積熱容量 Cv を抽出し、
原子参照エネルギーによる補正を行い、.xyz ファイルとして保存するスクリプト。

補正内容: Cv_corr = Cv_raw - Σ(n_i * Cv_atom_i)
"""

import os
import logging
import urllib.request
import numpy as np
from torch_geometric.datasets import QM9

# ==============================================================
# 1. 設定
# ==============================================================

OUTPUT_DIR = "/home/users/uchiyama/QM9_Cv/qm9_xyz_Cv"
DATA_DIR = "./data/QM9_raw"

# atomref.txt のURL
ATOMREF_URL = 'https://springernature.figshare.com/ndownloader/files/3195395'
ATOMREF_FILENAME = 'atomref.txt'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================
# 2. 関数定義
# ==============================================================

def download_file(url, save_path):
    """ファイルをダウンロード"""
    if not os.path.exists(save_path):
        logging.info(f"Downloading {os.path.basename(save_path)} ...")
        urllib.request.urlretrieve(url, filename=save_path)
    else:
        logging.info(f"File exists: {os.path.basename(save_path)}")

def load_atomref_cv(datadir):
    """
    atomref.txt を読み込み、Cv の原子参照値辞書を返す。
    形式: Element(0) ZPVE(1) U0(2) U(3) H(4) G(5) Cv(6)
    戻り値: {'H': 2.981..., 'C': 2.981...}
    """
    path = os.path.join(datadir, ATOMREF_FILENAME)
    
    # atomref.txt の列インデックス (Cvは6番目: 最後の列)
    TARGET_COL_IDX = 6
    
    atom_ref = {}
    
    with open(path, 'r') as f:
        for line in f:
            cols = line.split()
            # ヘッダー行などをスキップ
            if len(cols) < 7: continue
            
            elem = cols[0]
            if elem in ['H', 'C', 'N', 'O', 'F']:
                try:
                    val = float(cols[TARGET_COL_IDX])
                    atom_ref[elem] = val
                except ValueError:
                    pass
                    
    logging.info(f"Atom references loaded (Cv): {atom_ref}")
    return atom_ref

# ==============================================================
# 3. データセットをロード
# ==============================================================

print("🔹 Loading QM9 dataset...")
dataset = QM9(root='data/QM9')
print(f"✓ Loaded {len(dataset)} molecules.")

# ==============================================================
# 4. atomref.txt をダウンロード
# ==============================================================

os.makedirs(DATA_DIR, exist_ok=True)
ref_path = os.path.join(DATA_DIR, ATOMREF_FILENAME)
download_file(ATOMREF_URL, ref_path)

# 原子参照値を読み込み
ATOM_REF_CV = load_atomref_cv(DATA_DIR)

# ==============================================================
# 5. 物性インデックス（Cv）を特定
# ==============================================================

# torch_geometric.datasets.QM9 のラベル順序:
# ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', ...]
CV_INDEX = 11  # Cvは12番目（インデックス11）

# 原子番号から元素記号への変換
ATOMIC_NUM_TO_SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}

# ==============================================================
# 6. 処理と保存
# ==============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0

for idx, data in enumerate(dataset):
    # --- 1. 生のCvを取得 ---
    cv_raw = data.y[0, CV_INDEX].item()  # cal/mol·K
    
    # --- 2. 原子情報の取得 ---
    atomic_numbers = data.z.numpy()  # 原子番号
    positions = data.pos.numpy()      # 座標 (N, 3)
    
    # --- 3. 原子参照値の合計を計算 ---
    cv_ref_sum = 0.0
    atoms_data = []
    
    for i, z in enumerate(atomic_numbers):
        element = ATOMIC_NUM_TO_SYMBOL.get(int(z), 'X')
        
        if element in ATOM_REF_CV:
            cv_ref_sum += ATOM_REF_CV[element]
        else:
            logging.warning(f"Unknown element {element} (Z={z}) in molecule {idx}")
        
        x, y, z_coord = positions[i]
        atoms_data.append((element, x, y, z_coord))
    
    # --- 4. 補正計算 ---
    cv_corrected = cv_raw - cv_ref_sum
    
    # --- 5. XYZファイルとして保存 ---
    out_filename = f"mol_{idx:06d}.xyz"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    
    num_atoms = len(atoms_data)
    
    with open(out_path, "w") as out_f:
        out_f.write(f"{num_atoms}\n")
        out_f.write(f"Cv={cv_corrected:.6f} cal/mol·K\n")
        
        for atom in atoms_data:
            out_f.write(f"{atom[0]:<2}  {atom[1]: .6f}  {atom[2]: .6f}  {atom[3]: .6f}\n")
    
    count += 1
    if count % 5000 == 0:
        logging.info(f"✓ {count} molecules written...")

logging.info(f"✅ 完了: {count}個の分子を保存しました。")
logging.info(f"保存先: {OUTPUT_DIR}")
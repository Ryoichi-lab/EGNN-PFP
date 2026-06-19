#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# MEC_v1ディレクトリをパスに追加（build_graph_utils.pyがある場所）
BASE_DIR = "/home/users/uchiyama/AEC_v6/MEC_v1"
sys.path.insert(0, BASE_DIR)

import numpy as np
import torch
from tqdm import tqdm
from build_graph_utils import build_node_features, build_edges, build_edges_fully_connected

from torch_geometric.data import Data
import re
import json

# ===== 絶対パス設定 =====
QM9_H_DIR = "/home/users/uchiyama/QM9_H"
XYZ_DIR = os.path.join(QM9_H_DIR, "qm9_xyz_h") # XYZファイルがあるフォルダ
DESCRIPTOR_B3LYP_PATH = "/home/users/uchiyama/QM9_dipole/pfp_descriptors_qm9_B3LYP.npz"

# 出力ディレクトリ（2種類）
OUTPUT_DIR_BASELINE = os.path.join(QM9_H_DIR, "graphs_H_qm9_B3LYP_baseline_perfect")
OUTPUT_DIR_PFP = os.path.join(QM9_H_DIR, "graphs_H_qm9_B3LYP_pfp_perfect")

os.makedirs(OUTPUT_DIR_BASELINE, exist_ok=True)
os.makedirs(OUTPUT_DIR_PFP, exist_ok=True)

print("="*60)
print("QM9データセット B3LYP/6-31G(2df,p) H (Electronic spatial extent) 用グラフ作成")
print("EGNN論文準拠（Anderson et al. 2019 split: 100k/18k/13k）")
print("="*60)

# ===== XYZファイルから H を読み取る関数 =====
def read_xyz_with_H(xyz_path):
    """
    XYZファイルから原子座標とHを読み取る
    2行目に 'H=...' が書かれている前提
    """
    try:
        with open(xyz_path, 'r') as f:
            lines = f.readlines()
        
        n_atoms = int(lines[0].strip())
        comment_line = lines[1].strip()
        
        # H=... を抽出
        match = re.search(r'\bh=([-\d\.Ee+]+)', comment_line)
        if match:
            H_val = float(match.group(1))
        else:
            return None, None, None
        
        atoms = []
        positions = []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            atoms.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        return atoms, np.array(positions), H_val
    
    except Exception as e:
        print(f"Error reading {xyz_path}: {e}")
        return None, None, None

# ===== ベースライン用グラフ作成 =====
def create_graph_from_xyz_baseline(xyz_path, mol_id, cutoff=None):  # cutoffをNoneに
    try:
        atoms, positions, H_val = read_xyz_with_H(xyz_path)
        
        if atoms is None or positions is None or H_val is None:
            return None
        
        element_to_atomic_num = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        atomic_numbers = [element_to_atomic_num.get(atom, 1) for atom in atoms]
        
        n_atoms = len(atoms)
        dummy_pfp = np.zeros((n_atoms, 256), dtype=float)

        node_features = build_node_features(positions, atomic_numbers, dummy_pfp)
        # 完全グラフ版を使用（カットオフなし）
        edge_index, edge_features = build_edges_fully_connected(positions, dummy_pfp)

        if len(edge_features) > 0:
            edge_features = [[ef[0]] for ef in edge_features]

        return Data(
            x=torch.FloatTensor(node_features),
            pos=torch.FloatTensor(positions),
            edge_index=torch.LongTensor(edge_index).T if len(edge_index) > 0 else torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.FloatTensor(edge_features) if len(edge_features) > 0 else torch.zeros((0, 1)),
            y=torch.FloatTensor([H_val]),
            mol_id=mol_id
        )
    except Exception as e:
        print(f"分子 {mol_id} の処理エラー: {e}")
        return None

# create_graph_from_xyz_pfp 関数内を修正
def create_graph_from_xyz_pfp(xyz_path, pfp_descriptors, mol_id, cutoff=None):  # cutoffをNoneに
    try:
        atoms, positions, H_val = read_xyz_with_H(xyz_path)
        
        if atoms is None or positions is None or H_val is None:
            return None
        
        element_to_atomic_num = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        atomic_numbers = [element_to_atomic_num.get(atom, 1) for atom in atoms]
        
        n_atoms = len(atoms)
        if pfp_descriptors.shape[0] != n_atoms:
            print(f"[警告] 原子数不一致: {mol_id}")
            return None

        node_features = build_node_features(positions, atomic_numbers, pfp_descriptors)
        # 完全グラフ版を使用（カットオフなし）
        edge_index, edge_features = build_edges_fully_connected(positions, pfp_descriptors)

        return Data(
            x=torch.FloatTensor(node_features),
            pos=torch.FloatTensor(positions),
            edge_index=torch.LongTensor(edge_index).T if len(edge_index) > 0 else torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.FloatTensor(edge_features) if len(edge_features) > 0 else torch.zeros((0, 4)),
            y=torch.FloatTensor([H_val]),
            mol_id=mol_id
        )
    except Exception as e:
        print(f"分子 {mol_id} の処理エラー: {e}")
        return None

# ===== グラフデータ作成ループ関数 =====
def create_molecular_graphs_baseline(mol_ids, xyz_dir, split_name):
    graph_data_list = []
    failed_molecules = []
    
    print(f"\n{split_name}セットのグラフデータ作成中（ベースライン）...")
    for mol_id in tqdm(mol_ids):
        xyz_path = os.path.join(xyz_dir, mol_id)
        
        if not os.path.exists(xyz_path):
            failed_molecules.append(mol_id)
            continue
        
        graph_data = create_graph_from_xyz_baseline(xyz_path=xyz_path, mol_id=mol_id)
        
        if graph_data is not None:
            graph_data_list.append(graph_data)
        else:
            failed_molecules.append(mol_id)
    
    print(f"{split_name}: {len(graph_data_list)}個成功, {len(failed_molecules)}個失敗")
    return graph_data_list

def create_molecular_graphs_pfp(mol_ids, descriptor_data, xyz_dir, split_name):
    graph_data_list = []
    failed_molecules = []
    
    print(f"\n{split_name}セットのグラフデータ作成中（PFP版）...")
    for mol_id in tqdm(mol_ids):
        xyz_path = os.path.join(xyz_dir, mol_id)
        
        if mol_id not in descriptor_data:
            failed_molecules.append(mol_id)
            continue
        
        pfp_descriptors = descriptor_data[mol_id]
        
        if not os.path.exists(xyz_path):
            failed_molecules.append(mol_id)
            continue
        
        graph_data = create_graph_from_xyz_pfp(
            xyz_path=xyz_path,
            pfp_descriptors=pfp_descriptors,
            mol_id=mol_id
        )
        
        if graph_data is not None:
            graph_data_list.append(graph_data)
        else:
            failed_molecules.append(mol_id)
    
    print(f"{split_name}: {len(graph_data_list)}個成功, {len(failed_molecules)}個失敗")
    return graph_data_list

# ===== PFP記述子ロード =====
print("\n🔹 B3LYP PFP記述子をロード中...")
if not os.path.exists(DESCRIPTOR_B3LYP_PATH):
    print(f"❌ エラー: PFP記述子ファイルが見つかりません: {DESCRIPTOR_B3LYP_PATH}")
    sys.exit(1)

descriptor_data = np.load(DESCRIPTOR_B3LYP_PATH, allow_pickle=True)
print(f"✓ {len(descriptor_data.keys())} molecules loaded")

# ===== XYZファイルリスト取得 =====
print(f"\n🔹 XYZファイルをスキャン中...")
if not os.path.exists(XYZ_DIR):
    print(f"❌ エラー: XYZディレクトリが見つかりません: {XYZ_DIR}")
    sys.exit(1)

xyz_files = sorted([f for f in os.listdir(XYZ_DIR) if f.endswith('.xyz')])
print(f"✓ Total XYZ files: {len(xyz_files)}")

# ===== データの整合性チェック =====
print("\n🔹 データの整合性チェック...")
xyz_set = set(xyz_files)
pfp_set = set(descriptor_data.keys())
common_mols = xyz_set & pfp_set

print(f"  XYZファイル数: {len(xyz_set)}")
print(f"  PFP記述子の分子数: {len(pfp_set)}")
print(f"  共通分子数: {len(common_mols)}")

mol_ids = sorted(list(common_mols))
print(f"✓ 使用する分子数: {len(mol_ids)}")

# ===== データ分割（EGNN論文 / Anderson et al. 2019 準拠） =====
print("\n🔹 データを分割中（EGNN論文準拠）...")
print("  分割: Train 100,000 / Val 18,000 / Test 13,000")

n_total = len(mol_ids)
n_train = 100000
n_val = 18000
n_test = 13000
n_egnn_total = n_train + n_val + n_test

# 分子数が足りない場合の安全策（比率で調整）
if n_total < n_egnn_total:
    print(f"⚠️  警告: 利用可能な分子数({n_total})がEGNN論文の総数({n_egnn_total})より少ない")
    print(f"    利用可能な分子数に合わせて比率を維持します")
    ratio_train = n_train / n_egnn_total
    ratio_val = n_val / n_egnn_total
    
    n_train_actual = int(n_total * ratio_train)
    n_val_actual = int(n_total * ratio_val)
    n_test_actual = n_total - n_train_actual - n_val_actual
else:
    n_train_actual = n_train
    n_val_actual = n_val
    n_test_actual = n_test

print(f"\n  実際の分割数:")
print(f"    Train: {n_train_actual} molecules")
print(f"    Val:   {n_val_actual} molecules")
print(f"    Test:  {n_test_actual} molecules")

# ランダムシード固定
np.random.seed(42)
torch.manual_seed(42)
np.random.shuffle(mol_ids)

train_mol_ids = mol_ids[:n_train_actual]
val_mol_ids = mol_ids[n_train_actual:n_train_actual + n_val_actual]
test_mol_ids = mol_ids[n_train_actual + n_val_actual:n_train_actual + n_val_actual + n_test_actual]

# ===== グラフ生成 =====
print("\n" + "="*60)
print("🔹 グラフ生成開始")
print("="*60)

print("\n【1/2】EGNNベースライン（PFPなし）グラフ生成中...")
train_graphs_baseline = create_molecular_graphs_baseline(train_mol_ids, XYZ_DIR, "H_Baseline_Train")
val_graphs_baseline = create_molecular_graphs_baseline(val_mol_ids, XYZ_DIR, "H_Baseline_Val")
test_graphs_baseline = create_molecular_graphs_baseline(test_mol_ids, XYZ_DIR, "H_Baseline_Test")

print("\n【2/2】EGNN×PFP グラフ生成中...")
train_graphs_pfp = create_molecular_graphs_pfp(train_mol_ids, descriptor_data, XYZ_DIR, "H_PFP_Train")
val_graphs_pfp = create_molecular_graphs_pfp(val_mol_ids, descriptor_data, XYZ_DIR, "H_PFP_Val")
test_graphs_pfp = create_molecular_graphs_pfp(test_mol_ids, descriptor_data, XYZ_DIR, "H_PFP_Test")

# ===== 統計情報収集 =====
all_H_baseline = [g.y.item() for g in train_graphs_baseline + val_graphs_baseline + test_graphs_baseline]
all_H_pfp = [g.y.item() for g in train_graphs_pfp + val_graphs_pfp + test_graphs_pfp]

all_H_baseline = np.array(all_H_baseline)
all_H_pfp = np.array(all_H_pfp)

# ===== グラフデータを保存 =====
print("\n🔹 グラフデータを保存中...")

torch.save(train_graphs_baseline, os.path.join(OUTPUT_DIR_BASELINE, "train_graphs.pt"))
torch.save(val_graphs_baseline, os.path.join(OUTPUT_DIR_BASELINE, "val_graphs.pt"))
torch.save(test_graphs_baseline, os.path.join(OUTPUT_DIR_BASELINE, "test_graphs.pt"))
print(f"✓ ベースライン保存完了: {OUTPUT_DIR_BASELINE}")

torch.save(train_graphs_pfp, os.path.join(OUTPUT_DIR_PFP, "train_graphs.pt"))
torch.save(val_graphs_pfp, os.path.join(OUTPUT_DIR_PFP, "val_graphs.pt"))
torch.save(test_graphs_pfp, os.path.join(OUTPUT_DIR_PFP, "test_graphs.pt"))
print(f"✓ PFP版保存完了: {OUTPUT_DIR_PFP}")

# ===== 統計情報保存 =====
stats_baseline = {
    'dataset': 'QM9',
    'functional': 'B3LYP/6-31G(2df,p)',
    'model_type': 'EGNN_baseline',
    'target': 'H (Electronic spatial extent)',
    'unit': 'Bohr²',
    'n_molecules_total': n_total,
    'split_config': {
        'train': n_train_actual,
        'val': n_val_actual,
        'test': n_test_actual,
        'split_method': 'EGNN/Cormorant: Train=100k, Val=18k, Test=13k',
        'source': 'Anderson et al. (2019)'
    },
    'random_seed': 42,
    'H_statistics': {
        'min': float(all_H_baseline.min()),
        'max': float(all_H_baseline.max()),
        'mean': float(all_H_baseline.mean()),
        'std': float(all_H_baseline.std())
    }
}

stats_pfp = {
    'dataset': 'QM9',
    'functional': 'B3LYP/6-31G(2df,p)',
    'model_type': 'EGNN_with_PFP',
    'target': 'H (Electronic spatial extent)',
    'unit': 'Bohr²',
    'n_molecules_total': n_total,
    'split_config': {
        'train': n_train_actual,
        'val': n_val_actual,
        'test': n_test_actual,
        'split_method': 'EGNN/Cormorant: Train=100k, Val=18k, Test=13k',
        'source': 'Anderson et al. (2019)'
    },
    'random_seed': 42,
    'H_statistics': {
        'min': float(all_H_pfp.min()),
        'max': float(all_H_pfp.max()),
        'mean': float(all_H_pfp.mean()),
        'std': float(all_H_pfp.std())
    }
}

with open(os.path.join(OUTPUT_DIR_BASELINE, 'dataset_stats.json'), 'w') as f:
    json.dump(stats_baseline, f, indent=2)

with open(os.path.join(OUTPUT_DIR_PFP, 'dataset_stats.json'), 'w') as f:
    json.dump(stats_pfp, f, indent=2)

print("\n" + "="*60)
print("✅ QM9 B3LYP H グラフ作成完了!")
print("="*60)
print(f"\n【ベースライン】")
print(f"  保存先: {OUTPUT_DIR_BASELINE}")
print(f"  H範囲: {stats_baseline['H_statistics']['min']:.4f} - {stats_baseline['H_statistics']['max']:.4f} Bohr²")
print(f"\n【EGNN×PFP】")
print(f"  保存先: {OUTPUT_DIR_PFP}")
print(f"  H範囲: {stats_pfp['H_statistics']['min']:.4f} - {stats_pfp['H_statistics']['max']:.4f} Bohr²")
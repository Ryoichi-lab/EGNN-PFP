#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import re
import json
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data

# ===== パス設定 (環境に合わせて変更してください) =====
BASE_DIR = "/home/users/uchiyama/AEC_v6/MEC_v1"
sys.path.insert(0, BASE_DIR)

from build_graph_utils import build_node_features, build_edges

# ===== 絶対パス設定 =====
QM9_CV_DIR = "/home/users/uchiyama/QM9_Cv"

# ★ここを前処理スクリプトの出力先ディレクトリ名と一致させる★
# 前処理スクリプトでは "qm9_xyz_Cv" (または "qm9_xyz_Cv_corrected") になっているはず
XYZ_DIR = os.path.join(QM9_CV_DIR, "qm9_xyz_Cv") 

DESCRIPTOR_B3LYP_PATH = "/home/users/uchiyama/QM9_dipole/pfp_descriptors_qm9_B3LYP.npz"

# 出力ディレクトリ
OUTPUT_DIR_BASELINE = os.path.join(QM9_CV_DIR, "graphs_Cv_qm9_B3LYP_baseline_v2")
OUTPUT_DIR_PFP = os.path.join(QM9_CV_DIR, "graphs_Cv_qm9_B3LYP_pfp_v2")

os.makedirs(OUTPUT_DIR_BASELINE, exist_ok=True)
os.makedirs(OUTPUT_DIR_PFP, exist_ok=True)

print("="*60)
print("QM9データセット B3LYP/6-31G(2df,p) 定積熱容量 Cv 用グラフ作成")
print("EGNN論文準拠（Anderson et al. 2019 split）")
print("="*60)
print("\n目的:")
print("  1. EGNNベースライン（PFPなし）")
print("  2. EGNN×PFP（PFP記述子あり）")
print("  の性能比較（ターゲット: Cv）")
print(f"\n参照ディレクトリ: {XYZ_DIR}")


# ===== XYZファイルから Cv を読み取る関数 =====
def read_xyz_with_Cv(xyz_path):
    """
    XYZファイルから原子座標と定積モル熱容量 Cv を読み取る
    2行目に 'Cv=... ' が書かれている前提
    例: 'Cv=24.532 cal/mol·K' など
    """
    try:
        with open(xyz_path, 'r') as f:
            lines = f.readlines()
        
        # 1行目: 原子数
        n_atoms = int(lines[0].strip())
        
        # 2行目: Cv 情報
        comment_line = lines[1].strip()
        
        # "Cv=24.532 ..." の形式から値を抽出
        # \bCv= で "Cv=" にマッチさせる
        match = re.search(r'\bCv=([-\d\.Ee+]+)', comment_line)
        if match:
            Cv = float(match.group(1))
        else:
            print(f"[Warn] {os.path.basename(xyz_path)}: 'Cv=' not found in '{comment_line}'")
            return None, None, None
        
        # 3行目以降: 原子座標
        atoms = []
        positions = []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            atoms.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        return atoms, np.array(positions), Cv
    
    except Exception as e:
        print(f"Error reading {xyz_path}: {e}")
        return None, None, None


# ===== ベースライン用：PFPなしのグラフ作成関数 =====
def create_graph_from_xyz_baseline(xyz_path, mol_id, cutoff=5.0):
    try:
        atoms, positions, Cv = read_xyz_with_Cv(xyz_path)
        
        if atoms is None or positions is None or Cv is None:
            return None
        
        # 原子番号リストを作成
        element_to_atomic_num = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        atomic_numbers = [element_to_atomic_num.get(atom, 1) for atom in atoms]
        
        n_atoms = len(atoms)
        dummy_pfp = np.zeros((n_atoms, 256), dtype=float)

        node_features = build_node_features(positions, atomic_numbers, dummy_pfp)
        edge_index, edge_features = build_edges(positions, dummy_pfp, cutoff=cutoff)

        if len(edge_features) > 0:
            edge_features = [[ef[0]] for ef in edge_features] # 距離のみ

        return Data(
            x=torch.FloatTensor(node_features),
            pos=torch.FloatTensor(positions),
            edge_index=torch.LongTensor(edge_index).T if len(edge_index) > 0 else torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.FloatTensor(edge_features) if len(edge_features) > 0 else torch.zeros((0, 1)),
            y=torch.FloatTensor([Cv]),
            mol_id=mol_id
        )
    except Exception as e:
        print(f"分子 {mol_id} の処理エラー: {e}")
        return None

# ===== PFP版：グラフ作成関数 =====
def create_graph_from_xyz_pfp(xyz_path, pfp_descriptors, mol_id, cutoff=5.0):
    try:
        atoms, positions, Cv = read_xyz_with_Cv(xyz_path)
        
        if atoms is None or positions is None or Cv is None:
            return None
        
        element_to_atomic_num = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        atomic_numbers = [element_to_atomic_num.get(atom, 1) for atom in atoms]
        
        n_atoms = len(atoms)
        if pfp_descriptors.shape[0] != n_atoms:
            print(f"[警告] 原子数不一致: {mol_id} XYZ={n_atoms}, PFP={pfp_descriptors.shape[0]}")
            return None

        node_features = build_node_features(positions, atomic_numbers, pfp_descriptors)
        edge_index, edge_features = build_edges(positions, pfp_descriptors, cutoff=cutoff)

        return Data(
            x=torch.FloatTensor(node_features),
            pos=torch.FloatTensor(positions),
            edge_index=torch.LongTensor(edge_index).T if len(edge_index) > 0 else torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.FloatTensor(edge_features) if len(edge_features) > 0 else torch.zeros((0, 4)),
            y=torch.FloatTensor([Cv]),
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
        
        graph_data = create_graph_from_xyz_baseline(
            xyz_path=xyz_path,
            mol_id=mol_id
        )
        
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


# =============================================================================
# メイン処理
# =============================================================================

# 1. PFP記述子ロード
print("\n🔹 B3LYP PFP記述子をロード中...")
if not os.path.exists(DESCRIPTOR_B3LYP_PATH):
    print(f"❌ エラー: PFP記述子ファイルが見つかりません: {DESCRIPTOR_B3LYP_PATH}")
    sys.exit(1)

descriptor_data = np.load(DESCRIPTOR_B3LYP_PATH, allow_pickle=True)
print(f"✓ {len(descriptor_data.keys())} molecules loaded")


# 2. XYZファイルリスト取得
print(f"\n🔹 XYZファイルをスキャン中... (Dir: {XYZ_DIR})")

if not os.path.exists(XYZ_DIR):
    print(f"❌ エラー: XYZディレクトリが見つかりません")
    sys.exit(1)

xyz_files = sorted([f for f in os.listdir(XYZ_DIR) if f.endswith('.xyz')])
print(f"✓ Total XYZ files: {len(xyz_files)}")


# 3. データの整合性チェック
print("\n🔹 データの整合性チェック...")
xyz_set = set(xyz_files)
pfp_set = set(descriptor_data.keys())
common_mols = xyz_set & pfp_set

print(f"  XYZファイル数: {len(xyz_set)}")
print(f"  PFP記述子の分子数: {len(pfp_set)}")
print(f"  共通分子数: {len(common_mols)}")

if len(common_mols) == 0:
    print(f"❌ エラー: 共通の分子がありません")
    sys.exit(1)

mol_ids = sorted(list(common_mols))
print(f"✓ 使用する分子数: {len(mol_ids)}")


# 4. データ分割（EGNN論文準拠）
print("\n🔹 データを分割中（EGNN論文準拠）...")
# 論文: Train=100k, Test=10%, Val=残り(約18k)
# ただしデータ総数が少ない場合は比率を維持して調整
n_total = len(mol_ids)
n_train_target = 100000
n_val_target = 18000
n_test_target = 13000
n_paper_total = n_train_target + n_val_target + n_test_target

if n_total < n_paper_total:
    print(f"⚠️  データ総数({n_total})がEGNN論文の想定({n_paper_total})より少ないため、比率で調整します。")
    ratio_train = n_train_target / n_paper_total
    ratio_val = n_val_target / n_paper_total
    
    n_train_actual = int(n_total * ratio_train)
    n_val_actual = int(n_total * ratio_val)
    n_test_actual = n_total - n_train_actual - n_val_actual
else:
    n_train_actual = n_train_target
    n_val_actual = n_val_target
    n_test_actual = n_test_target # 残りをTestにする場合はここを調整

print(f"  Split Plan: Train={n_train_actual}, Val={n_val_actual}, Test={n_test_actual}")

np.random.seed(42)
torch.manual_seed(42)
np.random.shuffle(mol_ids)

train_mol_ids = mol_ids[:n_train_actual]
val_mol_ids = mol_ids[n_train_actual:n_train_actual + n_val_actual]
test_mol_ids = mol_ids[n_train_actual + n_val_actual:n_train_actual + n_val_actual + n_test_actual]


# 5. グラフ生成
print("\n" + "="*60)
print("🔹 グラフ生成開始 (Target: Cv)")
print("="*60)

# ベースライン
print("\n【1/2】EGNNベースライン（PFPなし）グラフ生成中...")
train_graphs_baseline = create_molecular_graphs_baseline(train_mol_ids, XYZ_DIR, "Cv_Baseline_Train")
val_graphs_baseline = create_molecular_graphs_baseline(val_mol_ids, XYZ_DIR, "Cv_Baseline_Val")
test_graphs_baseline = create_molecular_graphs_baseline(test_mol_ids, XYZ_DIR, "Cv_Baseline_Test")

# PFP版
print("\n【2/2】EGNN×PFP グラフ生成中...")
train_graphs_pfp = create_molecular_graphs_pfp(train_mol_ids, descriptor_data, XYZ_DIR, "Cv_PFP_Train")
val_graphs_pfp = create_molecular_graphs_pfp(val_mol_ids, descriptor_data, XYZ_DIR, "Cv_PFP_Val")
test_graphs_pfp = create_molecular_graphs_pfp(test_mol_ids, descriptor_data, XYZ_DIR, "Cv_PFP_Test")


# 6. 保存と統計情報
print("\n🔹 グラフデータを保存中...")

# Baseline Save
torch.save(train_graphs_baseline, os.path.join(OUTPUT_DIR_BASELINE, "train_graphs.pt"))
torch.save(val_graphs_baseline, os.path.join(OUTPUT_DIR_BASELINE, "val_graphs.pt"))
torch.save(test_graphs_baseline, os.path.join(OUTPUT_DIR_BASELINE, "test_graphs.pt"))

# PFP Save
torch.save(train_graphs_pfp, os.path.join(OUTPUT_DIR_PFP, "train_graphs.pt"))
torch.save(val_graphs_pfp, os.path.join(OUTPUT_DIR_PFP, "val_graphs.pt"))
torch.save(test_graphs_pfp, os.path.join(OUTPUT_DIR_PFP, "test_graphs.pt"))

# 統計情報
all_Cv_baseline = np.array([g.y.item() for g in train_graphs_baseline + val_graphs_baseline + test_graphs_baseline])
all_Cv_pfp = np.array([g.y.item() for g in train_graphs_pfp + val_graphs_pfp + test_graphs_pfp])

stats_baseline = {
    'dataset': 'QM9',
    'target': 'Cv',
    'unit': 'cal/mol·K',
    'n_total': len(all_Cv_baseline),
    'stats': {
        'min': float(all_Cv_baseline.min()),
        'max': float(all_Cv_baseline.max()),
        'mean': float(all_Cv_baseline.mean()),
        'std': float(all_Cv_baseline.std())
    }
}

stats_pfp = {
    'dataset': 'QM9',
    'target': 'Cv',
    'unit': 'cal/mol·K',
    'n_total': len(all_Cv_pfp),
    'stats': {
        'min': float(all_Cv_pfp.min()),
        'max': float(all_Cv_pfp.max()),
        'mean': float(all_Cv_pfp.mean()),
        'std': float(all_Cv_pfp.std())
    }
}

with open(os.path.join(OUTPUT_DIR_BASELINE, 'dataset_stats.json'), 'w') as f:
    json.dump(stats_baseline, f, indent=2)

with open(os.path.join(OUTPUT_DIR_PFP, 'dataset_stats.json'), 'w') as f:
    json.dump(stats_pfp, f, indent=2)

print("\n" + "="*60)
print("✅ QM9 B3LYP Cv グラフ作成完了!")
print("="*60)
print(f"\n【ベースライン】 保存先: {OUTPUT_DIR_BASELINE}")
print(f"  Train: {len(train_graphs_baseline)}")
print(f"  Val:   {len(val_graphs_baseline)}")
print(f"  Test:  {len(test_graphs_baseline)}")

print(f"\n【EGNN×PFP】 保存先: {OUTPUT_DIR_PFP}")
print(f"  Train: {len(train_graphs_pfp)}")
print(f"  Val:   {len(val_graphs_pfp)}")
print(f"  Test:  {len(test_graphs_pfp)}")
print("="*60)
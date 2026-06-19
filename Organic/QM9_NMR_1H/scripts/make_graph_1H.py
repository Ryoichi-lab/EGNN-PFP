#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QM9NMR データセット用グラフ作成スクリプト（¹H NMR）
- ノード特徴: PFP記述子(256) + one-hot元素(5) + 距離特徴(4) = 265次元
- エッジ: 完全グラフ（カットオフなし）
- エッジ特徴: [距離, cos類似度, L2距離, 距離×cos類似度] = 4次元
- ターゲット: 水素原子の¹Hシールディング定数（気相, 1列目）
- データ分割: 5-fold KFold（分子単位）各foldを独立保存
  学習時に任意のfoldを結合して使用可能
  例: 80/20 → train=fold0-3, val/test=fold4
  先行研究 (matlantis-contrib) の GroupKFold に準拠
"""

import os
import json
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from tqdm import tqdm

# ===== パス設定 =====
BASE_DIR     = os.path.expanduser("~/qm9nmr")
NMR_FILE     = os.path.join(BASE_DIR, "main/SI_DFT_NMR.txt")
XYZ_DIR      = os.path.join(BASE_DIR, "split")
DESC_FILE    = os.path.join(BASE_DIR, "QM9NMR_PFP_descriptor.npz")
OUTPUT_DIR   = os.path.join(BASE_DIR, "EGNN_PFP/graphs/1H")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_ELEMENT = 'H'
TARGET_Z       = 1
SOLVENT_COL    = 0   # 0=Gas, 1=CCl4, 2=THF, 3=Acetone, 4=Methanol, 5=DMSO
N_FOLDS        = 5    # 各foldを独立保存、学習時に結合
RANDOM_SEED    = 42

ELEM_TO_IDX  = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}  # one-hot用
N_ELEM       = 5

print("=" * 60)
print("QM9NMR ¹H NMR グラフ作成")
print(f"  溶媒: Gas phase (col={SOLVENT_COL})")
print(f"  完全グラフ × 全原子ノード (H, C, N, O, F)")
print("=" * 60)


# ===== NMRデータ読み込み =====
def parse_nmr_file(filepath, target_element, solvent_col):
    """
    SI_DFT_NMR.txt をパース。
    返り値: {mol_key: {'shielding': np.array}}
    """
    print(f"\n[1/4] NMRデータをパース中: {filepath}")
    mol_data = {}
    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    mol_count = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.isdigit():
            n_atoms   = int(line)
            mol_name  = lines[i + 1].strip()
            mol_key   = mol_name + ".xyz"
            target_shielding = []
            for j in range(i + 2, i + 2 + n_atoms):
                parts = lines[j].split()
                elem  = parts[0]
                if elem == target_element:
                    vals = [float(v) for v in parts[1:]]
                    target_shielding.append(vals[solvent_col])
            if target_shielding:
                mol_data[mol_key] = {
                    'shielding': np.array(target_shielding, dtype=np.float32)
                }
            i += 2 + n_atoms
            mol_count += 1
        else:
            i += 1

    print(f"  総分子数: {mol_count}")
    print(f"  {target_element}原子を含む分子数: {len(mol_data)}")
    return mol_data


# ===== ユーティリティ =====
def read_xyz(xyz_path):
    with open(xyz_path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    atoms, positions = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        atoms.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.array(positions, dtype=np.float32)


def build_node_features(positions, atoms, pfp_desc):
    """PFP(256) + one-hot元素(5) + 距離特徴(4) = 265次元"""
    center = positions.mean(axis=0)
    feats  = []
    for i in range(len(positions)):
        dists = [np.linalg.norm(positions[i] - positions[j])
                 for j in range(len(positions)) if j != i]
        oh = np.zeros(N_ELEM, dtype=np.float32)
        oh[ELEM_TO_IDX.get(atoms[i], 0)] = 1.0
        feat = np.concatenate([
            pfp_desc[i],
            oh,
            [np.linalg.norm(positions[i] - center),
             np.min(dists) if dists else 0.0,
             np.mean(dists) if dists else 0.0,
             float(np.sum(np.array(dists) < 3.0))]
        ])
        feats.append(feat)
    return np.array(feats, dtype=np.float32)


def build_edges_fully_connected(positions, pfp_desc):
    """完全グラフ（双方向エッジ）、エッジ特徴4次元"""
    edge_index, edge_feats = [], []
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist   = float(np.linalg.norm(positions[i] - positions[j]))
            dot    = float(np.dot(pfp_desc[i], pfp_desc[j]))
            norm_i = float(np.linalg.norm(pfp_desc[i]))
            norm_j = float(np.linalg.norm(pfp_desc[j]))
            cos_sim = dot / (norm_i * norm_j + 1e-8)
            l2_dist = float(np.linalg.norm(pfp_desc[i] - pfp_desc[j]))
            feat = [dist, cos_sim, l2_dist, dist * cos_sim]
            edge_index.extend([[i, j], [j, i]])
            edge_feats.extend([feat, feat])
    return edge_index, edge_feats


def create_graph(mol_key, xyz_dir, pfp_data, nmr_data):
    xyz_path = os.path.join(xyz_dir, mol_key)
    atoms, positions = read_xyz(xyz_path)
    pfp_desc = pfp_data[mol_key]

    if len(atoms) != pfp_desc.shape[0]:
        return None

    target_indices = [i for i, a in enumerate(atoms) if a == TARGET_ELEMENT]
    if len(target_indices) == 0:
        return None
    if len(target_indices) != len(nmr_data[mol_key]['shielding']):
        return None

    node_feats = build_node_features(positions, atoms, pfp_desc)
    edge_index, edge_feats = build_edges_fully_connected(positions, pfp_desc)

    # ターゲット: H原子のみ値、それ以外は -1
    y = torch.full((len(atoms),), -1.0, dtype=torch.float32)
    for idx, atom_i in enumerate(target_indices):
        y[atom_i] = float(nmr_data[mol_key]['shielding'][idx])

    # マスク
    mask = torch.zeros(len(atoms), dtype=torch.bool)
    mask[target_indices] = True

    return Data(
        x          = torch.FloatTensor(node_feats),
        pos        = torch.FloatTensor(positions),
        edge_index = torch.LongTensor(edge_index).T if edge_index else torch.zeros((2, 0), dtype=torch.long),
        edge_attr  = torch.FloatTensor(edge_feats) if edge_feats else torch.zeros((0, 4)),
        y          = y,
        h_mask     = mask,   # ¹³Cはc_mask、¹Hはh_mask
        mol_id     = mol_key
    )


# ===== メイン処理 =====

# 1. NMRデータ読み込み
nmr_data = parse_nmr_file(NMR_FILE, TARGET_ELEMENT, SOLVENT_COL)

# 2. PFP記述子ロード
print(f"\n[2/4] PFP記述子をロード中: {DESC_FILE}")
pfp_data = np.load(DESC_FILE, allow_pickle=True)
pfp_keys = set(pfp_data.keys())
print(f"  PFP分子数: {len(pfp_keys)}")
sample_key = list(pfp_keys)[0]
print(f"  記述子次元: {pfp_data[sample_key].shape}")

# 3. 共通分子を抽出して5-fold分割（各fold独立保存）
print(f"\n[3/4] データ分割 (5-fold KFold: 各fold独立保存)")
common_keys = sorted(set(nmr_data.keys()) & pfp_keys)
print(f"  NMR ∩ PFP 共通分子数: {len(common_keys)}")

all_keys = np.array(common_keys)
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_keys = {}
for fold_idx, (_, held_idx) in enumerate(kf.split(all_keys)):
    fold_keys[fold_idx] = all_keys[held_idx].tolist()
    print(f"  Fold {fold_idx}: {len(fold_keys[fold_idx])} 分子")

# 4. グラフ生成（fold別）
def build_graphs(keys, split_name):
    graphs, failed = [], []
    for mol_key in tqdm(keys, desc=split_name):
        g = create_graph(mol_key, XYZ_DIR, pfp_data, nmr_data)
        if g is not None:
            graphs.append(g)
        else:
            failed.append(mol_key)
    print(f"  {split_name}: {len(graphs)} 成功 / {len(failed)} 失敗")
    return graphs

print(f"\n[4/4] グラフ生成中...")
fold_graphs = {}
for fold_idx in range(N_FOLDS):
    fold_graphs[fold_idx] = build_graphs(fold_keys[fold_idx], f"Fold {fold_idx}")

# 5. fold別に保存
for fold_idx in range(N_FOLDS):
    torch.save(fold_graphs[fold_idx],
               os.path.join(OUTPUT_DIR, f"fold_{fold_idx}_graphs.pt"))

# 6. 統計情報
all_graphs    = [g for fold in fold_graphs.values() for g in fold]
total_target  = sum(g.h_mask.sum().item() for g in all_graphs)
all_shielding = np.concatenate([g.y[g.h_mask].numpy() for g in all_graphs])

stats = {
    'target':           '1H_NMR_shielding',
    'element':          TARGET_ELEMENT,
    'solvent':          'Gas phase',
    'solvent_col':      SOLVENT_COL,
    'graph_type':       'fully_connected',
    'node_feature_dim': int(fold_graphs[0][0].x.shape[1]),
    'edge_feature_dim': 4,
    'n_common_molecules': len(common_keys),
    'fold_sizes':       {str(i): len(fold_graphs[i]) for i in range(N_FOLDS)},
    'total_target_atoms': int(total_target),
    'split_method':     f'KFold(n_splits={N_FOLDS}, shuffle=True)',
    'n_folds':          N_FOLDS,
    'random_seed':      RANDOM_SEED,
    'usage':            '80/20: train=fold0+1+2+3, val/test=fold4  |  60/20/20: train=fold0+1+2, val=fold3, test=fold4',
    'shielding_stats': {
        'min':  float(all_shielding.min()),
        'max':  float(all_shielding.max()),
        'mean': float(all_shielding.mean()),
        'std':  float(all_shielding.std()),
    }
}
with open(os.path.join(OUTPUT_DIR, "dataset_stats.json"), 'w') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"完了！ -> {OUTPUT_DIR}")
for fold_idx in range(N_FOLDS):
    print(f"  Fold {fold_idx}: {len(fold_graphs[fold_idx])} グラフ → fold_{fold_idx}_graphs.pt")
print(f"\n  [使用例]")
print(f"  80/20 (原著準拠): train=fold0+1+2+3, val/test=fold4")
print(f"  60/20/20:         train=fold0+1+2, val=fold3, test=fold4")
print(f"\n  ノード特徴次元: {stats['node_feature_dim']}")
print(f"  総 {TARGET_ELEMENT} 原子数（ターゲット）: {total_target}")
print(f"  シールディング統計: {all_shielding.mean():.2f} ± {all_shielding.std():.2f} ppm")
print(f"  範囲: {all_shielding.min():.2f} ~ {all_shielding.max():.2f} ppm")
print(f"{'='*60}")

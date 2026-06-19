#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case Study 外挿データ用グラフ作成スクリプト
- 5カテゴリ: 12drugs, 40drugs, GDB, PAH, pyrimidinone
- ノード特徴: PFP記述子(256) + one-hot元素(5) + 距離特徴(4) = 265次元
- エッジ: 完全グラフ（カットオフなし）
- エッジ特徴: [距離, cos類似度, L2距離, 距離×cos類似度] = 4次元
- c_mask (¹³C) と h_mask (¹H) を両方付与
- 出力: graphs/case_studies/{category}_graphs.pt (PyG Data のリスト)
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# =============================================
# パス設定
# =============================================

BASE_DIR   = os.path.expanduser("~/qm9nmr")
CASE_DIR   = os.path.join(BASE_DIR, "case_studies")
XYZ_DIR    = os.path.join(CASE_DIR, "split_xyz")
OUTPUT_DIR = os.path.join(BASE_DIR, "EGNN_PFP/graphs/case_studies")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORIES = {
    '12drugs':      'SI_12Drugs_DFT_NMR.txt',
    '40drugs':      'SI_40Drugs_DFT_NMR.txt',
    'GDB':          'SI_GDB10to17_DFT_NMR.txt',
    'PAH':          'SI_PAH_DFT_NMR.txt',
    'pyrimidinone': 'SI_pyrimidinone_DFT_NMR.txt',
}

ELEM_TO_IDX = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
N_ELEM      = 5

print("=" * 60)
print("Case Study グラフ作成（PFP版）")
print(f"  ノード特徴次元: 265 (PFP256 + one-hot5 + dist4)")
print(f"  エッジ特徴次元: 4")
print(f"  カテゴリ: {list(CATEGORIES.keys())}")
print("=" * 60)


# =============================================
# NMR ファイルパース
# =============================================

def parse_nmr_file(path):
    """
    SI_*_DFT_NMR.txt をパース。
    フォーマット:
        n_atoms
        mol_name
        ELEM  shielding
        ...
    返り値: {mol_name: {'atoms': [...], 'shielding': [...]}}
    """
    data = {}
    with open(path) as f:
        lines = [l.strip() for l in f.readlines()]
    i = 0
    while i < len(lines):
        if not lines[i]:
            i += 1
            continue
        if lines[i].isdigit():
            n        = int(lines[i])
            mol_name = lines[i + 1].strip()
            atoms, shielding = [], []
            for j in range(i + 2, i + 2 + n):
                parts = lines[j].split()
                atoms.append(parts[0])
                shielding.append(float(parts[1]))
            data[mol_name] = {'atoms': atoms, 'shielding': shielding}
            i += 2 + n
        else:
            i += 1
    return data


# =============================================
# XYZ パース
# =============================================

def parse_xyz(path):
    """xyz ファイルを読んで (atoms, positions) を返す"""
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    atoms, positions = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        atoms.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.array(positions, dtype=np.float32)


# =============================================
# ノード特徴量構築
# =============================================

def build_node_features(positions, atoms, pfp_desc):
    """
    PFP(256) + one-hot元素(5) + dist(4) = 265次元
    dist(4): [重心距離, 最近接距離, 平均距離, 配位数(3Å以内)]
    """
    center = positions.mean(axis=0)
    feats  = []
    for i in range(len(positions)):
        dists = [np.linalg.norm(positions[i] - positions[j])
                 for j in range(len(positions)) if j != i]
        oh = np.zeros(N_ELEM, dtype=np.float32)
        oh[ELEM_TO_IDX.get(atoms[i], 0)] = 1.0
        feat = np.concatenate([
            pfp_desc[i].astype(np.float32),
            oh,
            [np.linalg.norm(positions[i] - center),
             float(np.min(dists)) if dists else 0.0,
             float(np.mean(dists)) if dists else 0.0,
             float(np.sum(np.array(dists) < 3.0))]
        ])
        feats.append(feat)
    return np.array(feats, dtype=np.float32)


# =============================================
# エッジ特徴量構築（完全グラフ）
# =============================================

def build_edges_fully_connected(positions, pfp_desc):
    """
    エッジ特徴: [距離, cos類似度(PFP), L2距離(PFP), 距離×cos類似度] = 4次元
    """
    edge_index, edge_feats = [], []
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist    = float(np.linalg.norm(positions[i] - positions[j]))
            dot     = float(np.dot(pfp_desc[i], pfp_desc[j]))
            norm_i  = float(np.linalg.norm(pfp_desc[i]))
            norm_j  = float(np.linalg.norm(pfp_desc[j]))
            cos_sim = dot / (norm_i * norm_j + 1e-8)
            l2_dist = float(np.linalg.norm(pfp_desc[i] - pfp_desc[j]))
            feat = [dist, cos_sim, l2_dist, dist * cos_sim]
            edge_index.extend([[i, j], [j, i]])
            edge_feats.extend([feat, feat])
    return edge_index, edge_feats


# =============================================
# 1分子グラフ作成
# =============================================

def build_graph(mol_name, category, pfp_data, nmr_data):
    """
    1分子のグラフを PyG Data として作成。
    c_mask (C原子), h_mask (H原子) を両方付与。
    """
    xyz_key  = mol_name + '.xyz'
    xyz_path = os.path.join(XYZ_DIR, category, xyz_key)

    if not os.path.exists(xyz_path):
        return None, f"xyz not found: {xyz_path}"
    if xyz_key not in pfp_data:
        return None, f"PFP key not found: {xyz_key}"
    if mol_name not in nmr_data:
        return None, f"NMR key not found: {mol_name}"

    atoms, positions = parse_xyz(xyz_path)
    pfp_desc         = np.array(pfp_data[xyz_key], dtype=np.float32)
    nmr_info         = nmr_data[mol_name]

    if len(atoms) != pfp_desc.shape[0]:
        return None, f"atom count mismatch: xyz={len(atoms)}, pfp={pfp_desc.shape[0]}"
    if len(atoms) != len(nmr_info['atoms']):
        return None, f"atom count mismatch: xyz={len(atoms)}, nmr={len(nmr_info['atoms'])}"

    # 未知元素チェック
    unknown = [a for a in atoms if a not in ELEM_TO_IDX]
    if unknown:
        return None, f"unknown elements: {set(unknown)}"

    node_feats = build_node_features(positions, atoms, pfp_desc)
    edge_index_list, edge_feats = build_edges_fully_connected(positions, pfp_desc)

    y         = torch.tensor(nmr_info['shielding'], dtype=torch.float32)
    c_mask    = torch.tensor([a == 'C' for a in atoms], dtype=torch.float32)
    h_mask    = torch.tensor([a == 'H' for a in atoms], dtype=torch.float32)
    edge_idx  = (torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
                 if edge_index_list else torch.zeros(2, 0, dtype=torch.long))
    edge_attr = (torch.tensor(edge_feats, dtype=torch.float32)
                 if edge_feats else torch.zeros(0, 4, dtype=torch.float32))

    data = Data(
        x          = torch.tensor(node_feats, dtype=torch.float32),
        pos        = torch.tensor(positions,  dtype=torch.float32),
        edge_index = edge_idx,
        edge_attr  = edge_attr,
        y          = y,
        c_mask     = c_mask,
        h_mask     = h_mask,
        mol_name   = mol_name,
        category   = category,
        n_atoms    = len(atoms),
    )
    return data, None


# =============================================
# メイン処理
# =============================================

stats = {}

for cat, nmr_fname in CATEGORIES.items():
    print(f"\n[{cat}] 処理中...")

    # --- PFP 記述子 ---
    npz_path = os.path.join(CASE_DIR, f"{cat}_NMR_PFP_descriptor.npz")
    if not os.path.exists(npz_path):
        print(f"  ERROR: npz not found: {npz_path}")
        continue
    pfp_data = np.load(npz_path, allow_pickle=True)
    print(f"  PFP: {len(pfp_data.files)} 分子  shape例: {pfp_data[pfp_data.files[0]].shape}")

    # --- NMR 正解値 ---
    nmr_path = os.path.join(CASE_DIR, cat, nmr_fname)
    if not os.path.exists(nmr_path):
        print(f"  ERROR: NMR file not found: {nmr_path}")
        continue
    nmr_data = parse_nmr_file(nmr_path)
    print(f"  NMR: {len(nmr_data)} 分子")

    # --- グラフ作成 ---
    graphs     = []
    skip_count = 0
    skip_reasons = {}

    mol_names = list(nmr_data.keys())
    for mol_name in tqdm(mol_names, desc=f"  {cat}"):
        g, err = build_graph(mol_name, cat, pfp_data, nmr_data)
        if g is None:
            skip_count += 1
            reason = err.split(':')[0] if err else 'unknown'
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
        graphs.append(g)

    # --- 保存 ---
    out_path = os.path.join(OUTPUT_DIR, f"{cat}_graphs.pt")
    torch.save(graphs, out_path)

    n_c_atoms = sum(int(g.c_mask.sum().item()) for g in graphs)
    n_h_atoms = sum(int(g.h_mask.sum().item()) for g in graphs)

    stats[cat] = {
        'n_mols':    len(graphs),
        'n_skipped': skip_count,
        'n_C_atoms': n_c_atoms,
        'n_H_atoms': n_h_atoms,
    }

    print(f"  保存: {out_path}")
    print(f"  成功: {len(graphs)} 分子  スキップ: {skip_count}")
    print(f"  C原子数: {n_c_atoms}  H原子数: {n_h_atoms}")
    if skip_reasons:
        for r, cnt in skip_reasons.items():
            print(f"    skip ({r}): {cnt}")
    if graphs:
        g0 = graphs[0]
        print(f"  サンプル [{g0.mol_name}]: {g0.n_atoms} atoms, "
              f"x={g0.x.shape}, edge_index={g0.edge_index.shape}, "
              f"edge_attr={g0.edge_attr.shape}")

# =============================================
# サマリー
# =============================================

print("\n" + "=" * 60)
print("  サマリー")
print("=" * 60)
print(f"{'Category':15s}  {'Mols':>5s}  {'C atoms':>8s}  {'H atoms':>8s}")
print("-" * 45)
for cat, s in stats.items():
    print(f"{cat:15s}  {s['n_mols']:5d}  {s['n_C_atoms']:8d}  {s['n_H_atoms']:8d}")
print(f"\n出力先: {OUTPUT_DIR}")
print("=" * 60)

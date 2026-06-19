#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EGNN×PFP (Metal_q) グラフに Bond Order (BO) を追加
"""

import torch
import os
import pickle
from tqdm import tqdm

BASE_DIR = "/home/users/uchiyama/tmQM_dipole"
INPUT_DIR = "/home/users/uchiyama/tmQM_dipole/graphs_tmQM_pfp_metalq2"
OUTPUT_DIR = "/home/users/uchiyama/tmQM_dipole/qgraphs_tmQM_pfp_metalq_bo2"
BOND_ORDER_FILE = os.path.join(BASE_DIR, "bond_orders_tmqm.pkl")

# 既存の出力ディレクトリを削除して再作成
import shutil
if os.path.exists(OUTPUT_DIR):
    print(f"⚠️  既存の {OUTPUT_DIR} を削除中...")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("🧩 EGNN×PFP (Metal_q) → BO追加版 変換スクリプト")
print("=" * 70)
print(f"入力元: {INPUT_DIR}")
print(f"出力先: {OUTPUT_DIR}")
print(f"結合次数データ: {BOND_ORDER_FILE}")

print("\n🔹 結合次数データをロード中...")
with open(BOND_ORDER_FILE, "rb") as f:
    bond_order_data = pickle.load(f)
print(f"✓ {len(bond_order_data)} molecules with bond orders")

def add_bond_order_to_graph(graph, bond_orders):
    """PFP版: edge_attr は 4次元 → 5次元に拡張"""
    if graph.edge_index.shape[1] == 0:
        graph.edge_attr = torch.zeros((0, 5), dtype=torch.float32)
        return graph

    edge_index = graph.edge_index.T.cpu().numpy()
    new_edge_attr = []

    for (src, dst), old_feat in zip(edge_index, graph.edge_attr):
        bo = bond_orders.get((int(src), int(dst)), 0.0)
        old_feat_list = old_feat.cpu().numpy().tolist()
        new_feat = old_feat_list + [bo]
        new_edge_attr.append(new_feat)

    graph.edge_attr = torch.tensor(new_edge_attr, dtype=torch.float32)
    return graph

splits = ["train", "val", "test"]

for split in splits:
    input_path = os.path.join(INPUT_DIR, f"{split}_graphs.pt")
    output_path = os.path.join(OUTPUT_DIR, f"{split}_graphs.pt")

    if not os.path.exists(input_path):
        print(f"⚠️ {input_path} が見つかりません")
        continue

    print(f"\n🔹 {split.upper()} セット処理中...")
    graphs = torch.load(input_path, weights_only=False)
    new_graphs = []

    success_count = 0
    skip_count = 0
    total_edges = 0
    bo_nonzero = 0

    for g in tqdm(graphs, desc=f"{split} graphs"):
        mol_id_raw = getattr(g, "mol_id", None)
        
        if mol_id_raw is None:
            # mol_idがない場合はゼロ埋めで追加
            if g.edge_index.shape[1] > 0:
                zeros = torch.zeros((g.edge_attr.shape[0], 1), dtype=torch.float32)
                g.edge_attr = torch.cat([g.edge_attr, zeros], dim=1)
            new_graphs.append(g)
            skip_count += 1
            continue
        
        # .xyz拡張子を除去
        mol_id_key = mol_id_raw.replace('.xyz', '') if mol_id_raw.endswith('.xyz') else mol_id_raw
        
        if mol_id_key not in bond_order_data:
            # BOデータがない場合はゼロ埋めで追加
            if g.edge_index.shape[1] > 0:
                zeros = torch.zeros((g.edge_attr.shape[0], 1), dtype=torch.float32)
                g.edge_attr = torch.cat([g.edge_attr, zeros], dim=1)
            new_graphs.append(g)
            skip_count += 1
            continue

        bond_orders = bond_order_data[mol_id_key]
        g_bo = add_bond_order_to_graph(g, bond_orders)
        new_graphs.append(g_bo)
        success_count += 1
        
        # 統計
        if g_bo.edge_attr.shape[0] > 0:
            total_edges += g_bo.edge_attr.shape[0]
            bo_nonzero += (g_bo.edge_attr[:, -1] > 0).sum().item()

    torch.save(new_graphs, output_path)
    print(f"✅ 保存完了: {output_path}")
    print(f"  Total graphs: {len(new_graphs)}")
    print(f"  BO追加成功: {success_count} ({success_count/len(new_graphs)*100:.1f}%)")
    print(f"  BOゼロ埋め: {skip_count} ({skip_count/len(new_graphs)*100:.1f}%)")
    print(f"  総エッジ数: {total_edges:,}")
    if total_edges > 0:
        print(f"  BO非ゼロエッジ: {bo_nonzero:,} ({bo_nonzero/total_edges*100:.1f}%)")
    
    # サンプル確認
    if len(new_graphs) > 0:
        sample = new_graphs[0]
        print(f"  ✅ edge_attr shape: {sample.edge_attr.shape} (期待: [N, 5])")
        if sample.edge_attr.shape[1] == 5 and sample.edge_attr.shape[0] > 0:
            bo_vals = sample.edge_attr[:, -1].numpy()
            print(f"     BO統計: min={bo_vals.min():.3f}, max={bo_vals.max():.3f}, mean={bo_vals.mean():.3f}")

print("\n✅ 完了！")
print("=" * 70)
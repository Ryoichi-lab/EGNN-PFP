#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Case Study 外挿性評価 - Baseline 単一分割版
training_13C_baseline/best_model.pth と training_1H_baseline/best_model.pth を使用
ノード特徴: one-hot元素(5) + 距離特徴(4) = 9次元（PFPなし）
エッジ特徴: [距離] = 1次元
5カテゴリ (12drugs, 40drugs, GDB, PAH, pyrimidinone) で ¹³C / ¹H MAE を評価
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
import os, json
from datetime import datetime
from sklearn.metrics import r2_score


# =============================================
# パス設定
# =============================================

BASE_DIR        = os.path.expanduser("~/qm9nmr")
CASE_DIR        = os.path.join(BASE_DIR, "case_studies")
XYZ_DIR         = os.path.join(CASE_DIR, "split_xyz")
MODEL_13C_PATH  = os.path.join(BASE_DIR, "EGNN_PFP/training_13C_baseline/best_model.pth")
MODEL_1H_PATH   = os.path.join(BASE_DIR, "EGNN_PFP/training_1H_baseline/best_model.pth")
OUTPUT_DIR      = os.path.join(BASE_DIR, "EGNN_PFP/case_study_results_baseline")

CATEGORIES = {
    '12drugs':      'SI_12Drugs_DFT_NMR.txt',
    '40drugs':      'SI_40Drugs_DFT_NMR.txt',
    'GDB':          'SI_GDB10to17_DFT_NMR.txt',
    'PAH':          'SI_PAH_DFT_NMR.txt',
    'pyrimidinone': 'SI_pyrimidinone_DFT_NMR.txt',
}

ELEM_TO_IDX = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
N_ELEM      = 5


# =============================================
# データパース
# =============================================

def parse_xyz(path):
    """xyz ファイルを読んで (atoms, positions) を返す"""
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    atoms, positions = [], []
    for line in lines[2:2+n]:
        parts = line.split()
        atoms.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.array(positions, dtype=np.float32)


def parse_nmr_file(path):
    """SI_*_DFT_NMR.txt を読んで {mol_name: {atoms, shielding}} を返す"""
    data = {}
    with open(path) as f:
        lines = [l.strip() for l in f.readlines()]
    i = 0
    while i < len(lines):
        if not lines[i]:
            i += 1; continue
        if lines[i].isdigit():
            n = int(lines[i])
            mol_name = lines[i+1].strip()
            atoms, shielding = [], []
            for j in range(i+2, i+2+n):
                parts = lines[j].split()
                atoms.append(parts[0])
                shielding.append(float(parts[1]))
            data[mol_name] = {'atoms': atoms, 'shielding': shielding}
            i += 2 + n
        else:
            i += 1
    return data


# =============================================
# グラフ構築（Baseline: PFPなし）
# =============================================

def build_node_features(positions, atoms):
    """one-hot元素(5) + 距離特徴(4) = 9次元"""
    center = positions.mean(axis=0)
    feats  = []
    for i in range(len(positions)):
        dists = [np.linalg.norm(positions[i] - positions[j])
                 for j in range(len(positions)) if j != i]
        oh = np.zeros(N_ELEM, dtype=np.float32)
        oh[ELEM_TO_IDX.get(atoms[i], 0)] = 1.0
        feat = np.concatenate([
            oh,
            [np.linalg.norm(positions[i] - center),
             np.min(dists) if dists else 0.0,
             np.mean(dists) if dists else 0.0,
             float(np.sum(np.array(dists) < 3.0))]
        ])
        feats.append(feat)
    return np.array(feats, dtype=np.float32)


def build_edges_fully_connected(positions):
    """完全グラフ（双方向エッジ）、エッジ特徴1次元（距離のみ）"""
    edge_index, edge_feats = [], []
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            edge_index.extend([[i, j], [j, i]])
            edge_feats.extend([[dist], [dist]])
    return edge_index, edge_feats


def build_graph(mol_name, xyz_dir, category, nmr_data):
    """1分子のグラフを作成（Baseline版: PFPなし）"""
    xyz_key  = mol_name + '.xyz'
    xyz_path = os.path.join(xyz_dir, category, xyz_key)
    if not os.path.exists(xyz_path):
        return None
    if mol_name not in nmr_data:
        return None

    atoms, positions = parse_xyz(xyz_path)
    nmr_info         = nmr_data[mol_name]

    if len(atoms) != len(nmr_info['atoms']):
        return None

    node_feats              = build_node_features(positions, atoms)
    edge_index_list, edge_feats = build_edges_fully_connected(positions)

    y         = torch.tensor(nmr_info['shielding'], dtype=torch.float32)
    c_mask    = torch.tensor([a == 'C' for a in atoms], dtype=torch.float32)
    h_mask    = torch.tensor([a == 'H' for a in atoms], dtype=torch.float32)
    edge_idx  = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous() \
                if edge_index_list else torch.zeros(2, 0, dtype=torch.long)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32) \
                if edge_feats else torch.zeros(0, 1)

    return Data(
        x          = torch.tensor(node_feats, dtype=torch.float32),
        pos        = torch.tensor(positions,  dtype=torch.float32),
        edge_index = edge_idx,
        edge_attr  = edge_attr,
        y          = y,
        c_mask     = c_mask,
        h_mask     = h_mask,
        mol_name   = mol_name,
    )


# =============================================
# EGNN モデル定義（訓練スクリプトと同一）
# =============================================

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    seg_exp = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result  = data.new_full(result_shape, 0)
    count   = data.new_full(result_shape, 0)
    result.scatter_add_(0, seg_exp, data)
    count.scatter_add_(0, seg_exp, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL_mask(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
                 nodes_attr_dim=0, act_fn=nn.SiLU(), recurrent=True,
                 coords_weight=1.0, attention=False):
        super().__init__()
        self.recurrent = recurrent
        self.attention = attention
        self.edge_mlp  = nn.Sequential(
            nn.Linear(input_nf * 2 + 1 + edges_in_d, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf), act_fn)
        self.node_mlp  = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_attr_dim, hidden_nf), act_fn,
            nn.Linear(hidden_nf, output_nf))
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        inp = torch.cat([source, target, radial], dim=1) if edge_attr is None \
              else torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(inp)
        if self.attention:
            out = out * self.att_mlp(out)
        return out

    def node_model(self, x, edge_index, edge_feat, node_attr):
        row, _ = edge_index
        agg = unsorted_segment_mean(edge_feat, row, num_segments=x.size(0))
        agg = torch.cat([x, agg, node_attr], dim=1) if node_attr is not None \
              else torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        diff   = coord[row] - coord[col]
        radial = torch.sum(diff**2, dim=1, keepdim=True)
        return radial, diff

    def forward(self, h, edge_index, coord, node_mask, edge_mask,
                edge_attr=None, node_attr=None, n_nodes=None):
        row, col  = edge_index
        radial, _ = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        edge_feat = edge_feat * edge_mask
        h, _      = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


class EGNN_NMR(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu',
                 act_fn=nn.SiLU(), n_layers=7, coords_weight=1.0,
                 attention=False, node_attr=True):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device    = device
        self.n_layers  = n_layers
        self.node_attr = node_attr
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        n_node_attr    = in_node_nf if node_attr else 0
        for i in range(n_layers):
            self.add_module(f"gcl_{i}", E_GCL_mask(
                hidden_nf, hidden_nf, hidden_nf,
                edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr,
                act_fn=act_fn, recurrent=True,
                coords_weight=coords_weight, attention=attention))
        self.node_dec = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf))
        self.nmr_head = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn,
            nn.Linear(hidden_nf, 1))
        self.to(device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(self.n_layers):
            node_attr_in = h0 if self.node_attr else None
            h, _, _ = self._modules[f"gcl_{i}"](
                h, edges, x, node_mask, edge_mask,
                edge_attr=edge_attr, node_attr=node_attr_in, n_nodes=n_nodes)
        h    = self.node_dec(h) * node_mask
        pred = self.nmr_head(h)
        return pred.squeeze(-1)


# =============================================
# バッチ前処理（mask_attr で ¹³C / ¹H を切替）
# =============================================

def prepare_batch(batch, device, mask_attr='c_mask'):
    batch_size = batch.batch.max().item() + 1
    node_counts, max_nodes = [], 0
    for i in range(batch_size):
        n = (batch.batch == i).sum().item()
        node_counts.append(n)
        max_nodes = max(max_nodes, n)

    total_nodes = batch_size * max_nodes
    h0          = torch.zeros(total_nodes, batch.x.size(1), device=device)
    x           = torch.zeros(total_nodes, 3, device=device)
    node_mask   = torch.zeros(total_nodes, 1, device=device)
    target      = torch.zeros(total_nodes, device=device)
    target_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)

    for i, n in enumerate(node_counts):
        mask = (batch.batch == i)
        s    = i * max_nodes
        h0[s:s+n]          = batch.x[mask].to(device)
        x[s:s+n]           = batch.pos[mask].to(device)
        node_mask[s:s+n]   = 1.0
        target[s:s+n]      = batch.y[mask].to(device)
        target_mask[s:s+n] = getattr(batch, mask_attr)[mask].bool().to(device)

    edges_list, edge_attr_list, edge_mask_list = [], [], []
    for i in range(batch_size):
        g_mask  = (batch.batch == i)
        g_nodes = g_mask.nonzero(as_tuple=True)[0]
        e_mask  = g_mask[batch.edge_index[0]] & g_mask[batch.edge_index[1]]
        if e_mask.sum() == 0:
            continue
        g_edges = batch.edge_index[:, e_mask]
        g_eattr = batch.edge_attr[e_mask]
        remap   = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
        remap[g_nodes] = torch.arange(len(g_nodes), device=device) + i * max_nodes
        edges_list.append(remap[g_edges.to(device)])
        edge_attr_list.append(g_eattr.to(device))
        edge_mask_list.append(torch.ones(g_eattr.size(0), 1, device=device))

    if edges_list:
        edges     = torch.cat(edges_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)
        edge_mask = torch.cat(edge_mask_list, dim=0)
    else:
        edges     = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_attr = torch.zeros(0, batch.edge_attr.size(1), device=device)
        edge_mask = torch.zeros(0, 1, device=device)

    return h0, x, edges, edge_attr, node_mask, edge_mask, max_nodes, target, target_mask


# =============================================
# 推論
# =============================================

def run_inference(model, graphs, device, mask_attr, batch_size=16):
    """グラフリストに対して推論し (preds, targets) を返す"""
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, target, tmask = \
                prepare_batch(batch, device, mask_attr)
            pred = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
            if torch.isnan(pred).any():
                continue
            all_preds.append(pred[tmask].cpu().numpy())
            all_targets.append(target[tmask].cpu().numpy())
    if not all_preds:
        return np.array([]), np.array([])
    return np.concatenate(all_preds), np.concatenate(all_targets)


# =============================================
# プロット
# =============================================

def plot_scatter(preds, targets, mae, r2, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(targets, preds, alpha=0.5, s=20, edgecolors='none')
    lo = min(targets.min(), preds.min()); hi = max(targets.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2)
    ax.set_xlabel('True NMR Shielding [ppm]', fontsize=12)
    ax.set_ylabel('Predicted NMR Shielding [ppm]', fontsize=12)
    ax.set_title(f'{title}\nMAE={mae:.4f} ppm  R²={r2:.4f}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    err = preds - targets
    ax.hist(err, bins=40, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', ls='--', lw=2)
    ax.set_xlabel('Error [ppm]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Error Distribution\nMean={err.mean():.4f}  Std={err.std():.4f}', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =============================================
# メイン
# =============================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── モデルロード ──
    def load_model(path, in_node_nf, in_edge_nf):
        model = EGNN_NMR(in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=128,
                         device=device, act_fn=nn.SiLU(), n_layers=7,
                         attention=True, node_attr=True).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"  Loaded: {path}  (epoch={ckpt.get('epoch','?')}  val_mae={ckpt.get('val_mae',float('nan')):.4f})")
        return model

    print("\n[1] モデルロード (Baseline: in_node_nf=9, in_edge_nf=1)...")
    model_13c = load_model(MODEL_13C_PATH, in_node_nf=9, in_edge_nf=1)
    model_1h  = load_model(MODEL_1H_PATH,  in_node_nf=9, in_edge_nf=1)

    all_results = {}

    for nucleus, model, mask_attr, outlier_thr in [
        ('13C', model_13c, 'c_mask', -50.0),
        ('1H',  model_1h,  'h_mask',   0.0),
    ]:
        print(f"\n{'='*70}")
        print(f"  ¹{nucleus} NMR 評価 (Baseline)")
        print(f"{'='*70}")
        nucleus_results = {}
        all_cat_preds, all_cat_targets = [], []

        for cat, nmr_fname in CATEGORIES.items():
            nmr_path = os.path.join(CASE_DIR, cat, nmr_fname)
            if not os.path.exists(nmr_path):
                print(f"  [{cat}] NMR file not found: {nmr_path}"); continue

            nmr_data = parse_nmr_file(nmr_path)
            all_graphs = []
            for mol_name in nmr_data.keys():
                g = build_graph(mol_name, XYZ_DIR, cat, nmr_data)
                if g is not None:
                    all_graphs.append(g)

            if not all_graphs:
                print(f"  [{cat}] グラフ構築失敗"); continue

            # 外れ値除去・ターゲット原子チェック
            graphs = []
            for g in all_graphs:
                mask = getattr(g, mask_attr).bool()
                if mask.sum() == 0:
                    continue
                if (g.y[mask] < outlier_thr).any():
                    continue
                graphs.append(g)

            if not graphs:
                print(f"  [{cat}] グラフなし"); continue

            # 推論
            preds, targets = run_inference(model, graphs, device, mask_attr)
            if len(preds) == 0:
                print(f"  [{cat}] 推論結果なし"); continue

            mae = float(np.mean(np.abs(preds - targets)))
            r2  = float(r2_score(targets, preds))
            nucleus_results[cat] = {'mae': mae, 'r2': r2, 'n_atoms': len(preds), 'n_mols': len(graphs)}
            all_cat_preds.append(preds)
            all_cat_targets.append(targets)

            print(f"  [{cat}] {len(graphs):3d} mol  {len(preds):6d} atoms  MAE={mae:.4f} ppm  R²={r2:.4f}")

            # プロット
            cat_dir = os.path.join(OUTPUT_DIR, nucleus)
            os.makedirs(cat_dir, exist_ok=True)
            plot_scatter(preds, targets, mae, r2,
                         f'¹{nucleus} NMR  {cat}  (Baseline Single Split)',
                         os.path.join(cat_dir, f'{cat}.png'))

        # 全カテゴリまとめ
        if all_cat_preds:
            all_p = np.concatenate(all_cat_preds)
            all_t = np.concatenate(all_cat_targets)
            overall_mae = float(np.mean(np.abs(all_p - all_t)))
            overall_r2  = float(r2_score(all_t, all_p))
            nucleus_results['overall'] = {'mae': overall_mae, 'r2': overall_r2,
                                          'n_atoms': len(all_p)}
            print(f"\n  [Overall] MAE={overall_mae:.4f} ppm  R²={overall_r2:.4f}  "
                  f"({len(all_p)} atoms)")
            plot_scatter(all_p, all_t, overall_mae, overall_r2,
                         f'¹{nucleus} NMR  All Categories  (Baseline Single Split)',
                         os.path.join(OUTPUT_DIR, nucleus, 'overall.png'))

        all_results[nucleus] = nucleus_results

    # JSON保存
    all_results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    all_results['version']   = 'baseline_single_split'
    out_json = os.path.join(OUTPUT_DIR, 'results.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    # サマリー表示
    print(f"\n{'='*70}")
    print("  結果サマリー（Baseline 単一分割）")
    print(f"{'='*70}")
    for nucleus in ['13C', '1H']:
        if nucleus not in all_results:
            continue
        print(f"\n  ¹{nucleus} NMR:")
        for cat, res in all_results[nucleus].items():
            if isinstance(res, dict):
                print(f"    {cat:15s}  MAE={res['mae']:.4f} ppm  R²={res['r2']:.4f}")
    print(f"\n  出力: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

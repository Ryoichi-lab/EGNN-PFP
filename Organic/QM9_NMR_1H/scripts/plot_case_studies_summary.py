#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Case Study 外挿性評価 - PFP Single Split
各カテゴリのy-yプロットをまとめたサマリーPNG図を作成
(13C と 1H それぞれ1枚ずつ)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch_geometric.data import Data, DataLoader
import os
from sklearn.metrics import r2_score


# =============================================
# パス設定
# =============================================

BASE_DIR        = os.path.expanduser("~/qm9nmr")
CASE_DIR        = os.path.join(BASE_DIR, "case_studies")
XYZ_DIR         = os.path.join(CASE_DIR, "split_xyz")
MODEL_13C_PATH  = os.path.join(BASE_DIR, "EGNN_PFP/training_13C/best_model.pth")
MODEL_1H_PATH   = os.path.join(BASE_DIR, "EGNN_PFP/training_1H/best_model.pth")
OUTPUT_DIR      = os.path.join(BASE_DIR, "EGNN_PFP/case_study_results_single")
GRAPH_CACHE_DIR = os.path.join(BASE_DIR, "EGNN_PFP/graphs/case_studies")

CATEGORIES = {
    '12drugs':      'SI_12Drugs_DFT_NMR.txt',
    '40drugs':      'SI_40Drugs_DFT_NMR.txt',
    'GDB':          'SI_GDB10to17_DFT_NMR.txt',
    'PAH':          'SI_PAH_DFT_NMR.txt',
    'pyrimidinone': 'SI_pyrimidinone_DFT_NMR.txt',
}
CAT_LABELS = {
    '12drugs':      '12 Drugs',
    '40drugs':      '40 Drugs',
    'GDB':          'GDB',
    'PAH':          'PAH',
    'pyrimidinone': 'Pyrimidinone',
}

ELEM_TO_IDX = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
N_ELEM      = 5


# =============================================
# データパース（キャッシュ優先）
# =============================================

def parse_nmr_file(path):
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
# EGNN モデル定義
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
# バッチ前処理
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
# サマリー図作成
# =============================================

def make_summary_figure(results_dict, nucleus, unit='ppm', save_path=None):
    """
    results_dict = {cat: {'preds': np.array, 'targets': np.array}, ...}
    ＋ 'overall': {同上}
    """
    cats = list(CATEGORIES.keys())  # 5 categories
    n_cats = len(cats)

    # 2行 × 3列 (最後は overall)
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    nucleus_sym = '¹³C' if nucleus == '13C' else '¹H'
    colors = {
        '12drugs':      '#2196F3',   # blue
        '40drugs':      '#4CAF50',   # green
        'GDB':          '#FF9800',   # orange
        'PAH':          '#F44336',   # red
        'pyrimidinone': '#9C27B0',   # purple
        'overall':      '#607D8B',   # blue-grey
    }

    positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    # 12drugs と 40drugs を左右入れ替え: 40drugs→(0,0), 12drugs→(0,1)
    panel_keys = ['40drugs', '12drugs'] + [c for c in cats if c not in ('12drugs', '40drugs')] + ['overall']

    for idx, (key, (row, col)) in enumerate(zip(panel_keys, positions)):
        ax = fig.add_subplot(gs[row, col])
        if key not in results_dict:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            continue

        preds   = results_dict[key]['preds']
        targets = results_dict[key]['targets']
        n_atoms = len(preds)
        n_mols  = results_dict[key].get('n_mols', '?')

        mae = float(np.mean(np.abs(preds - targets)))
        r2  = float(r2_score(targets, preds))

        color = colors.get(key, '#607D8B')

        # 範囲設定（自動スケーリング）
        pad = (targets.max() - targets.min()) * 0.05
        lo  = targets.min() - pad
        hi  = targets.max() + pad

        ax.scatter(targets, preds, alpha=0.45, s=12 if n_atoms > 500 else 25,
                   color=color, edgecolors='none', rasterized=True)
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, alpha=0.7)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')

        label = 'Overall (all)' if key == 'overall' else CAT_LABELS.get(key, key)
        mol_str = f'n={n_mols} mol, {n_atoms} atoms' if key != 'overall' else f'{n_atoms} atoms'

        ax.set_title(label, fontsize=13, fontweight='bold', pad=4)
        ax.set_xlabel(f'DFT {nucleus_sym} [ppm]', fontsize=10)
        ax.set_ylabel(f'Predicted {nucleus_sym} [ppm]', fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.25, lw=0.7)

        # テキスト注釈（左上に配置）
        textstr = f'$R^2$={r2:.4f}\nMAE={mae:.4f} ppm\n{mol_str}'
        ax.text(0.04, 0.96, textstr,
                transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.85, edgecolor=color, linewidth=1.2))

    fig.suptitle(f'EGNN-PFP  {nucleus_sym} NMR Chemical Shift Prediction\n'
                 f'Case Study External Test Sets (Single Split)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> 保存: {save_path}")


# =============================================
# メイン
# =============================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    def load_model(path, in_node_nf, in_edge_nf):
        model = EGNN_NMR(in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=128,
                         device=device, act_fn=nn.SiLU(), n_layers=7,
                         attention=True, node_attr=True).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"  Loaded: {path}  epoch={ckpt.get('epoch','?')}  val_mae={ckpt.get('val_mae',float('nan')):.4f}")
        return model

    print("\n[1] モデルロード...")
    model_13c = load_model(MODEL_13C_PATH, in_node_nf=265, in_edge_nf=4)
    model_1h  = load_model(MODEL_1H_PATH,  in_node_nf=265, in_edge_nf=4)

    for nucleus, model, mask_attr, outlier_thr in [
        ('13C', model_13c, 'c_mask', -50.0),
        ('1H',  model_1h,  'h_mask',   0.0),
    ]:
        print(f"\n{'='*60}")
        print(f"  ¹{nucleus} NMR 推論中...")
        print(f"{'='*60}")

        results = {}
        all_preds_global, all_targets_global = [], []

        for cat in CATEGORIES.keys():
            cache_path = os.path.join(GRAPH_CACHE_DIR, f"{cat}_graphs.pt")
            if os.path.exists(cache_path):
                all_graphs = torch.load(cache_path, map_location='cpu', weights_only=False)
                print(f"  [{cat}] キャッシュロード ({len(all_graphs)} mol)")
            else:
                print(f"  [{cat}] キャッシュなし → スキップ"); continue

            # 外れ値除去
            graphs = []
            for g in all_graphs:
                m = getattr(g, mask_attr).bool()
                if m.sum() == 0:
                    continue
                if (g.y[m] < outlier_thr).any():
                    continue
                graphs.append(g)

            if not graphs:
                print(f"  [{cat}] 有効グラフなし"); continue

            preds, targets = run_inference(model, graphs, device, mask_attr)
            if len(preds) == 0:
                continue

            mae = float(np.mean(np.abs(preds - targets)))
            r2  = float(r2_score(targets, preds))
            results[cat] = {'preds': preds, 'targets': targets,
                            'n_mols': len(graphs)}
            all_preds_global.append(preds)
            all_targets_global.append(targets)
            print(f"  [{cat}] {len(graphs):3d} mol  {len(preds):6d} atoms  MAE={mae:.4f}  R²={r2:.4f}")

        # Overall
        if all_preds_global:
            ap = np.concatenate(all_preds_global)
            at = np.concatenate(all_targets_global)
            results['overall'] = {'preds': ap, 'targets': at, 'n_mols': None}
            mae_all = float(np.mean(np.abs(ap - at)))
            r2_all  = float(r2_score(at, ap))
            print(f"  [Overall] {len(ap)} atoms  MAE={mae_all:.4f}  R²={r2_all:.4f}")

        # サマリー図保存
        save_path = os.path.join(OUTPUT_DIR, f'{nucleus}_summary_yy.png')
        make_summary_figure(results, nucleus, save_path=save_path)

    print(f"\n完了: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

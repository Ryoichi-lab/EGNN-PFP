#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Case Study: Baseline vs EGNN-PFP 比較スクリプト
5 カテゴリ (12drugs, 40drugs, GDB, PAH, pyrimidinone) について
¹H / ¹³C NMR 予測の重ね合わせプロットと指標比較を生成する。

出力: case_study_comparison/
  - 1H_scatter.png         : ¹H 散布図重ね合わせ（5カテゴリ × 2モデル）
  - 13C_scatter.png        : ¹³C 散布図重ね合わせ
  - metrics_comparison.png : MAE / R² バーチャート比較（全カテゴリ）
  - mae_improvement.png    : ΔMAE 改善量チャート（Baseline - PFP）
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import r2_score
from tqdm import tqdm


# =============================================
# パス設定
# =============================================

BASE_DIR       = os.path.expanduser("~/qm9nmr")
CASE_DIR       = os.path.join(BASE_DIR, "case_studies")
XYZ_DIR        = os.path.join(CASE_DIR, "split_xyz")
GRAPH_CACHE    = os.path.join(BASE_DIR, "EGNN_PFP/graphs/case_studies")
OUTPUT_DIR     = os.path.join(BASE_DIR, "EGNN_PFP/case_study_comparison")

MODEL_PATHS = {
    '1H':  {
        'pfp':      os.path.join(BASE_DIR, "EGNN_PFP/training_1H/best_model.pth"),
        'baseline': os.path.join(BASE_DIR, "EGNN_PFP/training_1H_baseline/best_model.pth"),
    },
    '13C': {
        'pfp':      os.path.join(BASE_DIR, "EGNN_PFP/training_13C/best_model.pth"),
        'baseline': os.path.join(BASE_DIR, "EGNN_PFP/training_13C_baseline/best_model.pth"),
    },
}

CATEGORIES = {
    '12drugs':      'SI_12Drugs_DFT_NMR.txt',
    '40drugs':      'SI_40Drugs_DFT_NMR.txt',
    'GDB':          'SI_GDB10to17_DFT_NMR.txt',
    'PAH':          'SI_PAH_DFT_NMR.txt',
    'pyrimidinone': 'SI_pyrimidinone_DFT_NMR.txt',
}

ELEM_TO_IDX = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
N_ELEM      = 5

COLOR_B = '#2196F3'   # Baseline: blue
COLOR_P = '#FF5722'   # PFP:      orange-red


# =============================================
# データパース
# =============================================

def parse_xyz(path):
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
    data = {}
    with open(path) as f:
        lines = [l.strip() for l in f.readlines()]
    i = 0
    while i < len(lines):
        if not lines[i]:
            i += 1; continue
        if lines[i].isdigit():
            n        = int(lines[i])
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
# Baseline グラフ構築（PFP なし: 9D + 1D）
# =============================================

def build_baseline_node_features(positions, atoms):
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


def build_baseline_edges(positions):
    edge_index, edge_feats = [], []
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            edge_index.extend([[i, j], [j, i]])
            edge_feats.extend([[dist], [dist]])
    return edge_index, edge_feats


def build_baseline_graph(mol_name, xyz_dir, category, nmr_data):
    xyz_path = os.path.join(xyz_dir, category, mol_name + '.xyz')
    if not os.path.exists(xyz_path) or mol_name not in nmr_data:
        return None
    atoms, positions = parse_xyz(xyz_path)
    nmr_info         = nmr_data[mol_name]
    if len(atoms) != len(nmr_info['atoms']):
        return None
    node_feats              = build_baseline_node_features(positions, atoms)
    edge_index_list, edge_feats = build_baseline_edges(positions)
    y        = torch.tensor(nmr_info['shielding'], dtype=torch.float32)
    c_mask   = torch.tensor([a == 'C' for a in atoms], dtype=torch.float32)
    h_mask   = torch.tensor([a == 'H' for a in atoms], dtype=torch.float32)
    edge_idx = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous() \
               if edge_index_list else torch.zeros(2, 0, dtype=torch.long)
    edge_a   = torch.tensor(edge_feats, dtype=torch.float32) \
               if edge_feats else torch.zeros(0, 1)
    return Data(x=torch.tensor(node_feats, dtype=torch.float32),
                pos=torch.tensor(positions, dtype=torch.float32),
                edge_index=edge_idx, edge_attr=edge_a,
                y=y, c_mask=c_mask, h_mask=h_mask, mol_name=mol_name)


# =============================================
# EGNN モデル
# =============================================

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    seg = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count  = data.new_full(result_shape, 0)
    result.scatter_add_(0, seg, data)
    count.scatter_add_(0, seg, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL_mask(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
                 nodes_attr_dim=0, act_fn=nn.SiLU(), recurrent=True,
                 coords_weight=1.0, attention=False):
        super().__init__()
        self.recurrent = recurrent
        self.attention = attention
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + 1 + edges_in_d, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf), act_fn)
        self.node_mlp = nn.Sequential(
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
# バッチ前処理 & 推論
# =============================================

def prepare_batch(batch, device, mask_attr):
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
        mask = (batch.batch == i); s = i * max_nodes
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
        if e_mask.sum() == 0: continue
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
            if torch.isnan(pred).any(): continue
            all_preds.append(pred[tmask].cpu().numpy())
            all_targets.append(target[tmask].cpu().numpy())
    if not all_preds:
        return np.array([]), np.array([])
    return np.concatenate(all_preds), np.concatenate(all_targets)


# =============================================
# プロット関数
# =============================================

def plot_scatter_overlay(ax, targets_b, preds_b, targets_p, preds_p,
                         mae_b, r2_b, mae_p, r2_p, cat, nucleus):
    """1カテゴリの重ね合わせ散布図"""
    all_vals = np.concatenate([targets_b, preds_b, targets_p, preds_p])
    lo, hi   = all_vals.min(), all_vals.max()
    margin   = (hi - lo) * 0.05 if (hi - lo) > 0 else 1.0

    ax.scatter(targets_b, preds_b, s=18, alpha=0.65, color=COLOR_B,
               label='Baseline', edgecolors='none', zorder=3)
    ax.scatter(targets_p, preds_p, s=18, alpha=0.65, color=COLOR_P,
               label='EGNN-PFP', edgecolors='none', zorder=4)
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            'k--', lw=1.2, zorder=2)
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_aspect('equal', adjustable='box')

    ax.set_title(cat, fontsize=11, fontweight='bold')
    ax.set_xlabel('True [ppm]', fontsize=9)
    ax.set_ylabel('Predicted [ppm]', fontsize=9)

    txt = (f'B: MAE={mae_b:.3f}  R2={r2_b:.3f}\n'
           f'P: MAE={mae_p:.3f}  R2={r2_p:.3f}')
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            fontsize=7.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.88, edgecolor='gray'))
    ax.grid(True, alpha=0.2)


def plot_metrics_bar(axes_mae_1h, axes_mae_13c, axes_r2_1h, axes_r2_13c,
                     results_all):
    """カテゴリ別 MAE / R² バーチャート（4パネル）"""
    cats    = list(CATEGORIES.keys()) + ['overall']
    x       = np.arange(len(cats))
    width   = 0.35
    configs = [
        (axes_mae_1h,  '1H',  'mae', 'MAE [ppm]',  '1H MAE by Category'),
        (axes_mae_13c, '13C', 'mae', 'MAE [ppm]',  '13C MAE by Category'),
        (axes_r2_1h,   '1H',  'r2',  'R2',          '1H R2 by Category'),
        (axes_r2_13c,  '13C', 'r2',  'R2',          '13C R2 by Category'),
    ]
    for ax, nucleus, metric, ylabel, title in configs:
        res = results_all[nucleus]
        vals_b = [res[c]['baseline'][metric] if c in res else 0.0 for c in cats]
        vals_p = [res[c]['pfp'][metric]      if c in res else 0.0 for c in cats]

        bars_b = ax.bar(x - width/2, vals_b, width, color=COLOR_B, alpha=0.85,
                        label='Baseline', edgecolor='white', linewidth=0.5)
        bars_p = ax.bar(x + width/2, vals_p, width, color=COLOR_P, alpha=0.85,
                        label='EGNN-PFP', edgecolor='white', linewidth=0.5)

        for bar in list(bars_b) + list(bars_p):
            h = bar.get_height()
            if abs(h) > 0.001:
                ax.text(bar.get_x() + bar.get_width()/2,
                        h * 1.02 if h > 0 else h - abs(h) * 0.1,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def plot_delta_mae(ax_1h, ax_13c, results_all):
    """ΔMAE = Baseline_MAE - PFP_MAE の棒グラフ（正 = PFP が優位）"""
    cats = list(CATEGORIES.keys()) + ['overall']
    x    = np.arange(len(cats))
    for ax, nucleus in [(ax_1h, '1H'), (ax_13c, '13C')]:
        res    = results_all[nucleus]
        deltas = [(res[c]['baseline']['mae'] - res[c]['pfp']['mae'])
                  if c in res else 0.0 for c in cats]
        colors = [COLOR_P if d > 0 else COLOR_B for d in deltas]
        bars   = ax.bar(x, deltas, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.axhline(0, color='black', lw=1.0, ls='--')
        for bar, d in zip(bars, deltas):
            if abs(d) > 0.001:
                ax.text(bar.get_x() + bar.get_width()/2,
                        d + (0.03 * max(abs(v) for v in deltas if abs(v) > 0) * np.sign(d)),
                        f'{d:+.3f}', ha='center', va='bottom' if d > 0 else 'top',
                        fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Baseline MAE - PFP MAE [ppm]', fontsize=10)
        ax.set_title(f'1{nucleus} MAE Improvement (+ = PFP better)', fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


# =============================================
# メイン
# =============================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── モデルロード ──
    def load_model(path, in_node_nf, in_edge_nf):
        m = EGNN_NMR(in_node_nf=in_node_nf, in_edge_nf=in_edge_nf,
                     hidden_nf=128, device=device, act_fn=nn.SiLU(),
                     n_layers=7, attention=True, node_attr=True).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        print(f"  Loaded: {os.path.basename(path)}  epoch={ckpt.get('epoch','?')}"
              f"  val_mae={ckpt.get('val_mae', float('nan')):.4f}")
        return m

    print("\n[1] モデルロード...")
    models = {
        '1H':  {
            'pfp':      load_model(MODEL_PATHS['1H']['pfp'],       265, 4),
            'baseline': load_model(MODEL_PATHS['1H']['baseline'],    9, 1),
        },
        '13C': {
            'pfp':      load_model(MODEL_PATHS['13C']['pfp'],      265, 4),
            'baseline': load_model(MODEL_PATHS['13C']['baseline'],   9, 1),
        },
    }

    # {nucleus: {cat: {baseline: {preds, targets, mae, r2}, pfp: ...}}}
    results_all = {'1H': {}, '13C': {}}

    nucleus_cfg = [
        ('1H',  'h_mask', 0.0),
        ('13C', 'c_mask', -50.0),
    ]

    print("\n[2] 推論...")
    for nucleus, mask_attr, outlier_thr in nucleus_cfg:
        print(f"\n  --- ¹{nucleus} ---")
        for cat, nmr_fname in CATEGORIES.items():
            print(f"  [{cat}]")

            # ── NMR データ読み込み ──
            nmr_path = os.path.join(CASE_DIR, cat, nmr_fname)
            if not os.path.exists(nmr_path):
                print(f"    NMR ファイルなし: {nmr_path}"); continue
            nmr_data = parse_nmr_file(nmr_path)

            # ── PFP グラフ（キャッシュ） ──
            cache_path = os.path.join(GRAPH_CACHE, f"{cat}_graphs.pt")
            if not os.path.exists(cache_path):
                print(f"    PFP キャッシュなし: {cache_path}"); continue
            pfp_graphs_all = torch.load(cache_path, map_location='cpu', weights_only=False)
            pfp_graphs = [g for g in pfp_graphs_all
                          if getattr(g, mask_attr).bool().sum() > 0
                          and not (g.y[getattr(g, mask_attr).bool()] < outlier_thr).any()]

            # ── Baseline グラフ（オンザフライ構築） ──
            mol_names = [g.mol_name for g in pfp_graphs]   # PFPと同じ分子セット
            baseline_graphs = []
            for mol_name in mol_names:
                g = build_baseline_graph(mol_name, XYZ_DIR, cat, nmr_data)
                if g is not None:
                    mask = getattr(g, mask_attr).bool()
                    if mask.sum() > 0 and not (g.y[mask] < outlier_thr).any():
                        baseline_graphs.append(g)

            n_pfp  = len(pfp_graphs)
            n_base = len(baseline_graphs)
            if n_pfp == 0 or n_base == 0:
                print(f"    グラフなし (PFP={n_pfp}, Baseline={n_base})"); continue

            # ── 推論 ──
            preds_p, targets_p = run_inference(
                models[nucleus]['pfp'],      pfp_graphs,      device, mask_attr)
            preds_b, targets_b = run_inference(
                models[nucleus]['baseline'], baseline_graphs, device, mask_attr)

            if len(preds_p) == 0 or len(preds_b) == 0:
                print(f"    推論結果なし"); continue

            mae_p = float(np.mean(np.abs(preds_p - targets_p)))
            r2_p  = float(r2_score(targets_p, preds_p))
            mae_b = float(np.mean(np.abs(preds_b - targets_b)))
            r2_b  = float(r2_score(targets_b, preds_b))
            print(f"    PFP:      {n_pfp:3d} mol  {len(preds_p):5d} atoms  "
                  f"MAE={mae_p:.4f}  R2={r2_p:.4f}")
            print(f"    Baseline: {n_base:3d} mol  {len(preds_b):5d} atoms  "
                  f"MAE={mae_b:.4f}  R2={r2_b:.4f}")

            results_all[nucleus][cat] = {
                'pfp':      {'preds': preds_p, 'targets': targets_p, 'mae': mae_p, 'r2': r2_p},
                'baseline': {'preds': preds_b, 'targets': targets_b, 'mae': mae_b, 'r2': r2_b},
            }

        # overall
        all_p_p, all_t_p, all_p_b, all_t_b = [], [], [], []
        for cat, res in results_all[nucleus].items():
            all_p_p.append(res['pfp']['preds']);      all_t_p.append(res['pfp']['targets'])
            all_p_b.append(res['baseline']['preds']); all_t_b.append(res['baseline']['targets'])
        if all_p_p:
            ap, at = np.concatenate(all_p_p), np.concatenate(all_t_p)
            bp, bt = np.concatenate(all_p_b), np.concatenate(all_t_b)
            results_all[nucleus]['overall'] = {
                'pfp':      {'preds': ap, 'targets': at,
                             'mae': float(np.mean(np.abs(ap - at))),
                             'r2':  float(r2_score(at, ap))},
                'baseline': {'preds': bp, 'targets': bt,
                             'mae': float(np.mean(np.abs(bp - bt))),
                             'r2':  float(r2_score(bt, bp))},
            }

    # =============================================
    # プロット 1: 散布図重ね合わせ（1H / 13C）
    # =============================================
    print("\n[3] プロット生成...")
    cats_plot = list(CATEGORIES.keys())   # overall は別図

    for nucleus, mask_attr, _ in nucleus_cfg:
        if not results_all[nucleus]:
            continue
        n_cats = len(cats_plot)
        fig, axes = plt.subplots(1, n_cats, figsize=(4 * n_cats, 4.2))
        if n_cats == 1:
            axes = [axes]

        for ax, cat in zip(axes, cats_plot):
            if cat not in results_all[nucleus]:
                ax.set_visible(False); continue
            res = results_all[nucleus][cat]
            rb, rp = res['baseline'], res['pfp']
            plot_scatter_overlay(ax, rb['targets'], rb['preds'],
                                 rp['targets'], rp['preds'],
                                 rb['mae'], rb['r2'], rp['mae'], rp['r2'],
                                 cat, nucleus)

        # 共通凡例
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_B,
                       markersize=8, label='Baseline'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_P,
                       markersize=8, label='EGNN-PFP'),
            plt.Line2D([0], [0], color='k', ls='--', lw=1.2, label='y = x'),
        ]
        fig.legend(handles=handles, loc='upper center', ncol=3,
                   fontsize=10, bbox_to_anchor=(0.5, 1.01))
        fig.suptitle(f'1{nucleus} NMR Case Study: Baseline vs EGNN-PFP',
                     fontsize=13, fontweight='bold', y=1.05)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = os.path.join(OUTPUT_DIR, f'{nucleus}_scatter.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  保存: {out_path}")

    # =============================================
    # プロット 2: MAE / R² バーチャート（4パネル）
    # =============================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plot_metrics_bar(axes[0, 0], axes[1, 0], axes[0, 1], axes[1, 1], results_all)
    fig.suptitle('Case Study Accuracy: Baseline vs EGNN-PFP',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'metrics_comparison.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {out_path}")

    # =============================================
    # プロット 3: ΔMAE 改善量
    # =============================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_delta_mae(ax1, ax2, results_all)
    fig.suptitle('Case Study MAE Improvement: Baseline - PFP (+ = PFP better)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'mae_improvement.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {out_path}")

    # =============================================
    # コンソールサマリー
    # =============================================
    cats_summary = list(CATEGORIES.keys()) + ['overall']
    print(f"\n{'='*72}")
    print("  Case Study 比較結果")
    print(f"{'='*72}")
    for nucleus, _, _ in nucleus_cfg:
        print(f"\n  ¹{nucleus} NMR:")
        print(f"  {'Category':<14} {'Baseline MAE':>14} {'PFP MAE':>12} "
              f"{'ΔMAE':>9} {'Baseline R2':>13} {'PFP R2':>9}")
        print("  " + "-" * 70)
        for cat in cats_summary:
            if cat not in results_all[nucleus]: continue
            rb = results_all[nucleus][cat]['baseline']
            rp = results_all[nucleus][cat]['pfp']
            delta = rb['mae'] - rp['mae']
            mark  = '(PFP+)' if delta > 0 else '(BASE+)'
            print(f"  {cat:<14} {rb['mae']:>14.4f} {rp['mae']:>12.4f} "
                  f"{delta:>+9.4f} {rb['r2']:>13.4f} {rp['r2']:>9.4f}  {mark}")
    print(f"\n  出力: {OUTPUT_DIR}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()

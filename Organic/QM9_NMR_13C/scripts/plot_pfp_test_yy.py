#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EGNN-PFP テストセット y-y プロット（¹H / ¹³C）
fold 4 (Val/Test set) に対する予測を散布図として出力する。

出力: comparison_results/
  - pfp_test_yy.png  : ¹H と ¹³C の y-y プロット（2パネル）
"""

import os, math
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm


# =============================================
# モデル定義（compare_baseline_pfp.py と同一）
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
            nn.Linear(input_nf * 2 + 1 + edges_in_d, hidden_nf),
            act_fn, nn.Linear(hidden_nf, hidden_nf), act_fn)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_attr_dim, hidden_nf),
            act_fn, nn.Linear(hidden_nf, output_nf))
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
        radial = torch.sum(diff ** 2, dim=1, keepdim=True)
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
        target_mask[s:s+n] = getattr(batch, mask_attr)[mask].to(device)
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


def evaluate(model, loader, device, mask_attr):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Eval", leave=False):
            batch = batch.to(device)
            h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, target, tmask = \
                prepare_batch(batch, device, mask_attr)
            pred = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
            if torch.isnan(pred).any(): continue
            all_preds.append(pred[tmask].cpu().numpy())
            all_targets.append(target[tmask].cpu().numpy())
    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return preds, targets


# =============================================
# メイン
# =============================================

def main():
    BASE      = os.path.expanduser("~/qm9nmr/EGNN_PFP")
    OUT_DIR   = os.path.join(BASE, "comparison_results")
    os.makedirs(OUT_DIR, exist_ok=True)

    VAL_FOLD = 4
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    configs = [
        {
            'nucleus':    '1H',
            'mask_attr':  'h_mask',
            'outlier_thr': 0.0,
            'graph_dir':  os.path.join(BASE, 'graphs/1H'),
            'model_path': os.path.join(BASE, 'training_1H/best_model.pth'),
            'in_node_nf': 265, 'in_edge_nf': 4,
            'color':      '#2196F3',
            'xlabel':     'DFT $^1$H NMR [ppm]',
            'ylabel':     'Predicted $^1$H NMR [ppm]',
        },
        {
            'nucleus':    '13C',
            'mask_attr':  'c_mask',
            'outlier_thr': -50.0,
            'graph_dir':  os.path.join(BASE, 'graphs/13C'),
            'model_path': os.path.join(BASE, 'training_13C/best_model.pth'),
            'in_node_nf': 265, 'in_edge_nf': 4,
            'color':      '#FF5722',
            'xlabel':     'DFT $^{13}$C NMR [ppm]',
            'ylabel':     'Predicted $^{13}$C NMR [ppm]',
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, cfg in zip(axes, configs):
        nucleus    = cfg['nucleus']
        mask_attr  = cfg['mask_attr']
        print(f"\n[{nucleus}] グラフ読み込み中...")

        # グラフロード & 外れ値除去
        graphs_all = torch.load(
            os.path.join(cfg['graph_dir'], f"fold_{VAL_FOLD}_graphs.pt"),
            weights_only=False, map_location='cpu')
        graphs = []
        for g in graphs_all:
            m = getattr(g, mask_attr).bool()
            if m.sum() > 0 and not (g.y[m] < cfg['outlier_thr']).any():
                graphs.append(g)
        n_mols = len(graphs)
        print(f"  {n_mols:,} mol")

        loader = DataLoader(graphs, batch_size=32, shuffle=False, num_workers=0)

        # モデルロード
        model = EGNN_NMR(
            in_node_nf=cfg['in_node_nf'], in_edge_nf=cfg['in_edge_nf'],
            hidden_nf=128, device=device, act_fn=nn.SiLU(),
            n_layers=7, attention=True, node_attr=True).to(device)
        ckpt = torch.load(cfg['model_path'], map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        best_epoch = ckpt['epoch']
        print(f"  Best epoch: {best_epoch}")

        # 推論
        print(f"  推論中...")
        preds, targets = evaluate(model, loader, device, mask_attr)
        n_atoms = len(preds)
        mae = float(np.mean(np.abs(preds - targets)))
        r2  = float(r2_score(targets, preds))
        print(f"  MAE={mae:.6f} ppm  R2={r2:.6f}  ({n_atoms:,} atoms)")

        # ── 散布図 ──
        MAX_PTS = 100_000
        if n_atoms > MAX_PTS:
            idx = np.random.default_rng(42).choice(n_atoms, MAX_PTS, replace=False)
            t_plot, p_plot = targets[idx], preds[idx]
        else:
            t_plot, p_plot = targets, preds

        ax.scatter(t_plot, p_plot, s=3, alpha=0.25, color=cfg['color'],
                   edgecolors='none', rasterized=True)

        lo = min(targets.min(), preds.min())
        hi = max(targets.max(), preds.max())
        margin = (hi - lo) * 0.02
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'k--', lw=1.5, label='y = x')
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel(cfg['xlabel'], fontsize=13)
        ax.set_ylabel(cfg['ylabel'], fontsize=13)
        ax.set_title(f'$^{{{nucleus}}}$NMR  Test Set (fold {VAL_FOLD})',
                     fontsize=13, fontweight='bold')

        info = (f'MAE = {mae:.4f} ppm\n'
                f'$R^2$ = {r2:.4f}\n'
                f'n = {n_mols:,} mol  /  {n_atoms:,} atoms')
        ax.text(0.04, 0.96, info, transform=ax.transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          alpha=0.88, edgecolor='gray'))
        ax.grid(True, alpha=0.2)

    fig.suptitle('EGNN-PFP  NMR Prediction  Test Set',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'pfp_test_yy.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()

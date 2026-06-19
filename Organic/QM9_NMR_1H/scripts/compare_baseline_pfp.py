#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline vs EGNN-PFP モデル精度比較スクリプト（単一分割）
¹H NMR と ¹³C NMR の両モデルを同一バリデーションセットで評価し、
重ね合わせプロットを生成する。

出力: comparison_results/
  - 1H_comparison.png   : ¹H 比較プロット（散布図・誤差分布・学習曲線）
  - 13C_comparison.png  : ¹³C 比較プロット
  - metrics_summary.png : MAE / R² バーチャート比較
"""

import os
import math
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score
from tqdm import tqdm


# =============================================
# Utility
# =============================================

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    seg = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count  = data.new_full(result_shape, 0)
    result.scatter_add_(0, seg, data)
    count.scatter_add_(0, seg, torch.ones_like(data))
    return result / count.clamp(min=1)


# =============================================
# E_GCL_mask Layer
# =============================================

class E_GCL_mask(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
                 nodes_attr_dim=0, act_fn=nn.SiLU(), recurrent=True,
                 coords_weight=1.0, attention=False):
        super().__init__()
        self.recurrent = recurrent
        self.attention = attention
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + 1 + edges_in_d, hidden_nf),
            act_fn, nn.Linear(hidden_nf, hidden_nf), act_fn
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_attr_dim, hidden_nf),
            act_fn, nn.Linear(hidden_nf, output_nf)
        )
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
        row, col   = edge_index
        radial, _  = self.coord2radial(edge_index, coord)
        edge_feat  = self.edge_model(h[row], h[col], radial, edge_attr)
        edge_feat  = edge_feat * edge_mask
        h, _       = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


# =============================================
# EGNN_NMR モデル（Baseline / PFP 共通）
# =============================================

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
                edges_in_d=in_edge_nf,
                nodes_attr_dim=n_node_attr,
                act_fn=act_fn, recurrent=True,
                coords_weight=coords_weight, attention=attention
            ))
        self.node_dec = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf)
        )
        self.nmr_head = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn,
            nn.Linear(hidden_nf, 1)
        )
        self.to(device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(self.n_layers):
            node_attr_in = h0 if self.node_attr else None
            h, _, _ = self._modules[f"gcl_{i}"](
                h, edges, x, node_mask, edge_mask,
                edge_attr=edge_attr, node_attr=node_attr_in, n_nodes=n_nodes
            )
        h    = self.node_dec(h) * node_mask
        pred = self.nmr_head(h)
        return pred.squeeze(-1)


# =============================================
# バッチ前処理（mask_attr で ¹H/¹³C 切り替え）
# =============================================

def prepare_batch(batch, device, mask_attr='h_mask'):
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
        target_mask[s:s+n] = getattr(batch, mask_attr)[mask].to(device)

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
# 評価関数
# =============================================

def evaluate(model, loader, device, mask_attr='h_mask'):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Eval", leave=False):
            batch = batch.to(device)
            h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, target, tmask = \
                prepare_batch(batch, device, mask_attr)
            pred = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
            if torch.isnan(pred).any():
                continue
            p, t = pred[tmask], target[tmask]
            all_preds.append(p.cpu().numpy())
            all_targets.append(t.cpu().numpy())
    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    mae = np.abs(preds - targets).mean()
    r2  = r2_score(targets, preds)
    return preds, targets, mae, r2


# =============================================
# グラフ読み込み・外れ値除去
# =============================================

def load_graphs(graph_dir, val_fold, mask_attr, outlier_thr):
    graphs = torch.load(
        os.path.join(graph_dir, f"fold_{val_fold}_graphs.pt"),
        weights_only=False, map_location='cpu'
    )
    ok, n_rem = [], 0
    for g in graphs:
        m = getattr(g, mask_attr).bool()
        if (g.y[m] < outlier_thr).any():
            n_rem += 1
        else:
            ok.append(g)
    print(f"    外れ値除去: {n_rem} mol  残: {len(ok):,} mol")
    return ok


# =============================================
# プロット関数
# =============================================

COLOR_B = '#2196F3'   # Baseline: blue
COLOR_P = '#FF5722'   # PFP:      orange-red

def plot_scatter_overlay(ax, targets_b, preds_b, targets_p, preds_p,
                         mae_b, r2_b, mae_p, r2_p, nucleus):
    """2モデルの散布図を重ね合わせ"""
    # ダウンサンプリング（点が多すぎる場合は間引く）
    MAX_PTS = 80_000
    if len(targets_b) > MAX_PTS:
        idx_b = np.random.choice(len(targets_b), MAX_PTS, replace=False)
        idx_p = np.random.choice(len(targets_p), MAX_PTS, replace=False)
    else:
        idx_b = np.arange(len(targets_b))
        idx_p = np.arange(len(targets_p))

    ax.scatter(targets_b[idx_b], preds_b[idx_b],
               alpha=0.15, s=4, color=COLOR_B, label='Baseline', edgecolors='none', rasterized=True)
    ax.scatter(targets_p[idx_p], preds_p[idx_p],
               alpha=0.15, s=4, color=COLOR_P, label='EGNN-PFP', edgecolors='none', rasterized=True)

    all_vals = np.concatenate([targets_b, preds_b, targets_p, preds_p])
    lo, hi   = all_vals.min(), all_vals.max()
    margin   = (hi - lo) * 0.02
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            'k--', lw=1.5, label='y = x')
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_xlabel(f'True ¹{nucleus} NMR [ppm]', fontsize=11)
    ax.set_ylabel(f'Predicted ¹{nucleus} NMR [ppm]', fontsize=11)
    ax.set_title(f'¹{nucleus} NMR Prediction (Val Set)', fontsize=12, fontweight='bold')

    # テキストボックスで指標を表示
    txt  = (f'Baseline: MAE={mae_b:.4f} ppm  R2={r2_b:.4f}\n'
            f'EGNN-PFP: MAE={mae_p:.4f} ppm  R2={r2_p:.4f}')
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax.legend(loc='lower right', fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.25)
    ax.set_aspect('equal', adjustable='box')


def plot_error_hist(ax, errors_b, errors_p, nucleus):
    """誤差分布の重ね合わせヒストグラム"""
    # 外れ値クリップ（±3σ以内）
    sigma_b = errors_b.std()
    sigma_p = errors_p.std()
    clip    = max(3 * sigma_b, 3 * sigma_p)
    eb = np.clip(errors_b, -clip, clip)
    ep = np.clip(errors_p, -clip, clip)

    bins = np.linspace(-clip, clip, 80)
    ax.hist(eb, bins=bins, alpha=0.55, color=COLOR_B, edgecolor='none',
            density=True, label=f'Baseline (μ={errors_b.mean():.4f}, σ={sigma_b:.4f})')
    ax.hist(ep, bins=bins, alpha=0.55, color=COLOR_P, edgecolor='none',
            density=True, label=f'EGNN-PFP (μ={errors_p.mean():.4f}, σ={sigma_p:.4f})')
    ax.axvline(0, color='black', ls='--', lw=1.5)
    ax.set_xlabel('Prediction Error [ppm]', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'¹{nucleus} Error Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.25)


def plot_learning_curve(ax, hist_b, hist_p, nucleus):
    """検証 MAE の学習曲線を重ね合わせ"""
    epochs_b = np.arange(1, len(hist_b['val_mae']) + 1)
    epochs_p = np.arange(1, len(hist_p['val_mae']) + 1)
    val_b    = np.array(hist_b['val_mae'])
    val_p    = np.array(hist_p['val_mae'])

    ax.plot(epochs_b, val_b, color=COLOR_B, lw=1.5, alpha=0.9, label='Baseline')
    ax.plot(epochs_p, val_p, color=COLOR_P, lw=1.5, alpha=0.9, label='EGNN-PFP')

    # 最良点にマーカー
    best_b = int(np.argmin(val_b))
    best_p = int(np.argmin(val_p))
    ax.scatter(epochs_b[best_b], val_b[best_b], color=COLOR_B, s=60, zorder=5,
               marker='*', edgecolors='black', linewidth=0.5)
    ax.scatter(epochs_p[best_p], val_p[best_p], color=COLOR_P, s=60, zorder=5,
               marker='*', edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Val MAE [ppm]', fontsize=11)
    ax.set_title(f'¹{nucleus} Learning Curve', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)


# =============================================
# メイン
# =============================================

def main():
    BASE      = os.path.expanduser("~/qm9nmr/EGNN_PFP")
    GRAPH_DIR = os.path.join(BASE, "graphs")
    OUT_DIR   = os.path.join(BASE, "comparison_results")
    os.makedirs(OUT_DIR, exist_ok=True)

    VAL_FOLD = 4
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 核種ごとの設定 ──
    configs = [
        {
            'nucleus':      '1H',
            'mask_attr':    'h_mask',
            'outlier_thr':  0.0,
            'baseline': {
                'graph_dir': os.path.join(GRAPH_DIR, '1H_baseline'),
                'model_dir': os.path.join(BASE, 'training_1H_baseline'),
                'in_node_nf': 9,  'in_edge_nf': 1,
            },
            'pfp': {
                'graph_dir': os.path.join(GRAPH_DIR, '1H'),
                'model_dir': os.path.join(BASE, 'training_1H'),
                'in_node_nf': 265, 'in_edge_nf': 4,
            },
        },
        {
            'nucleus':      '13C',
            'mask_attr':    'c_mask',
            'outlier_thr':  -50.0,
            'baseline': {
                'graph_dir': os.path.join(GRAPH_DIR, '13C_baseline'),
                'model_dir': os.path.join(BASE, 'training_13C_baseline'),
                'in_node_nf': 9,  'in_edge_nf': 1,
            },
            'pfp': {
                'graph_dir': os.path.join(GRAPH_DIR, '13C'),
                'model_dir': os.path.join(BASE, 'training_13C'),
                'in_node_nf': 265, 'in_edge_nf': 4,
            },
        },
    ]

    all_metrics = {}   # {nucleus: {baseline: {mae, r2}, pfp: {mae, r2}}}

    for cfg in configs:
        nucleus   = cfg['nucleus']
        mask_attr = cfg['mask_attr']
        print(f"\n{'='*60}")
        print(f"  ¹{nucleus} NMR 比較")
        print(f"{'='*60}")

        results = {}
        for model_type in ('baseline', 'pfp'):
            mc = cfg[model_type]
            print(f"\n  [{model_type.upper()}]")

            # --- グラフ読み込み ---
            print(f"  グラフ読み込み: {mc['graph_dir']}")
            val_graphs = load_graphs(mc['graph_dir'], VAL_FOLD, mask_attr, cfg['outlier_thr'])
            loader     = DataLoader(val_graphs, batch_size=32, shuffle=False, num_workers=0)

            # --- モデル構築・重みロード ---
            model = EGNN_NMR(
                in_node_nf   = mc['in_node_nf'],
                in_edge_nf   = mc['in_edge_nf'],
                hidden_nf    = 128,
                device       = device,
                act_fn       = nn.SiLU(),
                n_layers     = 7,
                attention    = True,
                node_attr    = True,
            ).to(device)

            ckpt_path = os.path.join(mc['model_dir'], 'best_model.pth')
            ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            history = ckpt['history']
            print(f"  モデルロード: epoch={ckpt['epoch']}  best_val_mae={ckpt['val_mae']:.6f}")

            # --- 推論 ---
            print(f"  推論中...")
            preds, targets, mae, r2 = evaluate(model, loader, device, mask_attr)
            errors = preds - targets
            print(f"  MAE={mae:.6f} ppm  R²={r2:.6f}")

            results[model_type] = {
                'preds': preds, 'targets': targets,
                'errors': errors, 'mae': mae, 'r2': r2,
                'history': history,
            }

        all_metrics[nucleus] = {k: {'mae': v['mae'], 'r2': v['r2']} for k, v in results.items()}

        # ── 比較プロット ──
        fig = plt.figure(figsize=(18, 5))
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        rb, rp = results['baseline'], results['pfp']

        plot_scatter_overlay(
            ax1,
            rb['targets'], rb['preds'],
            rp['targets'], rp['preds'],
            rb['mae'], rb['r2'], rp['mae'], rp['r2'],
            nucleus
        )
        plot_error_hist(ax2, rb['errors'], rp['errors'], nucleus)
        plot_learning_curve(ax3, rb['history'], rp['history'], nucleus)

        fig.suptitle(
            f"1{nucleus} NMR: Baseline vs EGNN-PFP  (Single Split, fold{VAL_FOLD} validation)",
            fontsize=13, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"{nucleus}_comparison.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n  プロット保存: {out_path}")

    # ── サマリーバーチャート ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    nuclei    = list(all_metrics.keys())
    x         = np.arange(len(nuclei))
    width     = 0.35

    for ax, metric, ylabel, title in zip(
        axes,
        ['mae', 'r2'],
        ['MAE [ppm]', 'R2'],
        ['MAE Comparison', 'R2 Comparison']
    ):
        vals_b = [all_metrics[n]['baseline'][metric] for n in nuclei]
        vals_p = [all_metrics[n]['pfp'][metric]      for n in nuclei]

        bars_b = ax.bar(x - width/2, vals_b, width, label='Baseline',
                        color=COLOR_B, alpha=0.85, edgecolor='white', linewidth=0.5)
        bars_p = ax.bar(x + width/2, vals_p, width, label='EGNN-PFP',
                        color=COLOR_P, alpha=0.85, edgecolor='white', linewidth=0.5)

        # 数値ラベル
        for bar in bars_b:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + h*0.01,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=9)
        for bar in bars_p:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + h*0.01,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        tick_labels = [f'¹{n} NMR' for n in nuclei]
        ax.set_xticklabels(tick_labels, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Baseline vs EGNN-PFP  Accuracy Summary (Single Split)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'metrics_summary.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nサマリープロット保存: {out_path}")

    # ── コンソール出力 ──
    print(f"\n{'='*60}")
    print("  最終比較結果")
    print(f"{'='*60}")
    print(f"{'Nucleus':<8} {'Model':<12} {'MAE [ppm]':>12} {'R²':>10}")
    print("-" * 46)
    for n in nuclei:
        for mt in ('baseline', 'pfp'):
            m = all_metrics[n][mt]
            label = 'Baseline' if mt == 'baseline' else 'EGNN-PFP'
            print(f"¹{n:<7} {label:<12} {m['mae']:>12.6f} {m['r2']:>10.6f}")
        print()
    print(f"出力ディレクトリ: {OUT_DIR}")


if __name__ == "__main__":
    main()

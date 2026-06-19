#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QM9 B3LYP 双極子モーメント予測 - テストセット誤差ヒストグラム比較
EGNN Baseline vs EGNN×PFP
"""

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy import stats

# =============================================
# Utility functions (両スクリプトから必要な関数をインポート)
# =============================================

def unsorted_segment_sum(data, segment_ids, num_segments):
    """セグメント単位の和"""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


# =============================================
# E_GCL_mask Layer
# =============================================

class E_GCL_mask(nn.Module):
    """E_GCL with masking support"""
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, 
                 nodes_attr_dim=0, act_fn=nn.SiLU(), recurrent=True, 
                 coords_weight=1.0, attention=False):
        super(E_GCL_mask, self).__init__()
        
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.epsilon = 1e-8
        
        edge_coords_nf = 1
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_attr_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )
        
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )
        
        self.act_fn = act_fn
    
    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        
        return out
    
    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        
        out = self.node_mlp(agg)
        
        if self.recurrent:
            out = x + out
        
        return out, agg
    
    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, dim=1, keepdim=True)
        
        norm = torch.sqrt(radial).detach() + self.epsilon
        coord_diff = coord_diff / norm
        
        return radial, coord_diff
    
    def forward(self, h, edge_index, coord, node_mask, edge_mask, 
                edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        edge_feat = edge_feat * edge_mask
        
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        
        return h, coord, edge_attr


# =============================================
# EGNN Model
# =============================================

class EGNN_Official(nn.Module):
    """EGNN公式実装"""
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', 
                 act_fn=nn.SiLU(), n_layers=7, coords_weight=1.0, 
                 attention=False, node_attr=True):
        super(EGNN_Official, self).__init__()
        
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.node_attr = node_attr
        
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        
        for i in range(n_layers):
            self.add_module(
                f"gcl_{i}",
                E_GCL_mask(
                    self.hidden_nf, 
                    self.hidden_nf, 
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    nodes_attr_dim=n_node_attr,
                    act_fn=act_fn,
                    recurrent=True,
                    coords_weight=coords_weight,
                    attention=attention
                )
            )
        
        self.node_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, self.hidden_nf)
        )
        
        self.graph_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, 1)
        )
        
        self.to(self.device)
    
    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        
        for i in range(self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules[f"gcl_{i}"](
                    h, edges, x, node_mask, edge_mask,
                    edge_attr=edge_attr,
                    node_attr=h0,
                    n_nodes=n_nodes
                )
            else:
                h, _, _ = self._modules[f"gcl_{i}"](
                    h, edges, x, node_mask, edge_mask,
                    edge_attr=edge_attr,
                    node_attr=None,
                    n_nodes=n_nodes
                )
        
        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        
        return pred.squeeze(1)


# =============================================
# データ前処理関数
# =============================================

def create_atom_onehot(batch, device):
    """原子番号からone-hot特徴を作成"""
    node_features = batch.x
    
    if node_features.size(1) == 5:
        return node_features.to(device)
    
    elif node_features.size(1) == 261:
        atomic_nums = node_features[:, 256].long()
        features = torch.zeros(atomic_nums.size(0), 5, device=device)
        atom_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        for i, z in enumerate(atomic_nums):
            z_val = z.item()
            if z_val in atom_map:
                features[i, atom_map[z_val]] = 1
            else:
                features[i, 0] = 1
        return features
    
    else:
        raise ValueError(f"Unexpected node feature dimension: {node_features.size(1)}")


def prepare_batch_pfp(batch, device):
    """PFP版のバッチ準備"""
    batch_size = batch.batch.max().item() + 1
    max_nodes = 0
    
    node_counts = []
    for i in range(batch_size):
        mask = (batch.batch == i)
        node_counts.append(mask.sum().item())
        max_nodes = max(max_nodes, mask.sum().item())
    
    total_nodes = batch_size * max_nodes
    
    h0 = torch.zeros(total_nodes, batch.x.size(1), device=device)
    x = torch.zeros(total_nodes, 3, device=device)
    node_mask = torch.zeros(total_nodes, 1, device=device)
    
    for i, n_nodes in enumerate(node_counts):
        mask = (batch.batch == i)
        start_idx = i * max_nodes
        
        h0[start_idx:start_idx + n_nodes] = batch.x[mask].to(device)
        x[start_idx:start_idx + n_nodes] = batch.pos[mask].to(device)
        node_mask[start_idx:start_idx + n_nodes] = 1.0
    
    edges_list = []
    edge_attr_list = []
    edge_mask_list = []
    
    for i in range(batch_size):
        graph_mask = (batch.batch == i)
        graph_nodes = graph_mask.nonzero(as_tuple=True)[0]
        
        edge_mask_graph = (graph_mask[batch.edge_index[0]] & 
                          graph_mask[batch.edge_index[1]])
        
        if edge_mask_graph.sum() > 0:
            graph_edges = batch.edge_index[:, edge_mask_graph]
            graph_edge_attr = batch.edge_attr[edge_mask_graph]
            
            old_to_new = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
            old_to_new[graph_nodes] = torch.arange(len(graph_nodes), device=device) + i * max_nodes
            
            new_edges = old_to_new[graph_edges.to(device)]
            
            edges_list.append(new_edges)
            edge_attr_list.append(graph_edge_attr.to(device))
            edge_mask_list.append(torch.ones(graph_edge_attr.size(0), 1, device=device))
    
    if edges_list:
        edges = torch.cat(edges_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)
        edge_mask = torch.cat(edge_mask_list, dim=0)
    else:
        edges = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_attr = torch.zeros(0, batch.edge_attr.size(1), device=device)
        edge_mask = torch.zeros(0, 1, device=device)
    
    return h0, x, edges, edge_attr, node_mask, edge_mask, max_nodes, batch.y.to(device)


def prepare_batch_baseline(batch, device):
    """Baseline版のバッチ準備"""
    batch_size = batch.batch.max().item() + 1
    max_nodes = 0
    
    node_counts = []
    for i in range(batch_size):
        mask = (batch.batch == i)
        node_counts.append(mask.sum().item())
        max_nodes = max(max_nodes, mask.sum().item())
    
    total_nodes = batch_size * max_nodes
    
    atom_features = create_atom_onehot(batch, device)
    
    h0 = torch.zeros(total_nodes, 5, device=device)
    x = torch.zeros(total_nodes, 3, device=device)
    node_mask = torch.zeros(total_nodes, 1, device=device)
    
    for i, n_nodes in enumerate(node_counts):
        mask = (batch.batch == i)
        start_idx = i * max_nodes
        
        h0[start_idx:start_idx + n_nodes] = atom_features[mask]
        x[start_idx:start_idx + n_nodes] = batch.pos[mask].to(device)
        node_mask[start_idx:start_idx + n_nodes] = 1.0
    
    edges_list = []
    edge_mask_list = []
    
    for i in range(batch_size):
        graph_mask = (batch.batch == i)
        graph_nodes = graph_mask.nonzero(as_tuple=True)[0]
        
        edge_mask_graph = (graph_mask[batch.edge_index[0]] & 
                          graph_mask[batch.edge_index[1]])
        
        if edge_mask_graph.sum() > 0:
            graph_edges = batch.edge_index[:, edge_mask_graph]
            
            old_to_new = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
            old_to_new[graph_nodes] = torch.arange(len(graph_nodes), device=device) + i * max_nodes
            
            new_edges = old_to_new[graph_edges.to(device)]
            
            edges_list.append(new_edges)
            edge_mask_list.append(torch.ones(new_edges.size(1), 1, device=device))
    
    if edges_list:
        edges = torch.cat(edges_list, dim=1)
        edge_mask = torch.cat(edge_mask_list, dim=0)
    else:
        edges = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_mask = torch.zeros(0, 1, device=device)
    
    edge_attr = None
    
    return h0, x, edges, edge_attr, node_mask, edge_mask, max_nodes, batch.y.to(device)


# =============================================
# テスト評価関数
# =============================================

def evaluate_model(model, loader, device, is_pfp=True):
    """モデルを評価して予測値とターゲットを取得"""
    model.eval()
    all_preds = []
    all_targets = []
    
    prepare_batch_fn = prepare_batch_pfp if is_pfp else prepare_batch_baseline
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)
            
            h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, target = \
                prepare_batch_fn(batch, device)
            
            pred = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    return predictions, targets

def plot_error_histograms_improved(baseline_errors, pfp_errors, output_dir):
    """改善版: 左右のプロットを個別ファイルにも出力"""
    
    # 統計量計算
    baseline_mae = np.abs(baseline_errors).mean()
    baseline_rmse = np.sqrt((baseline_errors ** 2).mean())
    baseline_std = baseline_errors.std()
    
    pfp_mae = np.abs(pfp_errors).mean()
    pfp_rmse = np.sqrt((pfp_errors ** 2).mean())
    pfp_std = pfp_errors.std()
    
    # 改善率
    mae_improvement = (baseline_mae - pfp_mae) / baseline_mae * 100
    rmse_improvement = (baseline_rmse - pfp_rmse) / baseline_rmse * 100
    
    # 99パーセンタイル範囲を計算
    baseline_p99 = np.percentile(np.abs(baseline_errors), 99)
    pfp_p99 = np.percentile(np.abs(pfp_errors), 99)
    xlim = max(baseline_p99, pfp_p99) * 1.1
    
    baseline_abs_errors = np.abs(baseline_errors)
    pfp_abs_errors = np.abs(pfp_errors)
    
    # ===================================================================
    # 1. 結合プロット（既存）
    # ===================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左: ヒストグラム
    ax1 = axes[0]
    bins = np.linspace(-xlim, xlim, 100)
    
    ax1.hist(baseline_errors, bins=bins, alpha=0.5, 
             color='cornflowerblue', edgecolor='steelblue', 
             linewidth=1.0, label=f'Baseline (MAE: {baseline_mae:.4f} D)', zorder=2)
    
    ax1.hist(pfp_errors, bins=bins, alpha=0.5, 
             color='mediumseagreen', edgecolor='seagreen', 
             linewidth=1.0, label=f'EGNN×PFP (MAE: {pfp_mae:.4f} D)', zorder=1)
    
    ax1.axvline(x=0, color='red', linestyle='-', linewidth=2.5, 
                zorder=10, alpha=0.9, label='Zero Error')
    
    ax1.axvline(x=baseline_errors.mean(), color='blue', 
                linestyle='--', linewidth=1.5, alpha=0.5, zorder=5)
    ax1.axvline(x=pfp_errors.mean(), color='green', 
                linestyle='--', linewidth=1.5, alpha=0.5, zorder=5)
    
    ax1.set_xlabel('Prediction Error (D)', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=15, fontweight='bold')
    ax1.set_xlim(-xlim, xlim)
    
    title_text = (f'Error Distribution Comparison\n'
                  f'Baseline: MAE={baseline_mae:.4f} D, Std={baseline_std:.4f} D\n'
                  f'PFP: MAE={pfp_mae:.4f} D, Std={pfp_std:.4f} D\n'
                  f'Improvement: MAE ↓{mae_improvement:.1f}%, RMSE ↓{rmse_improvement:.1f}%')
    ax1.set_title(title_text, fontsize=12, fontweight='bold', pad=15)
    
    legend = ax1.legend(fontsize=11, loc='upper left', 
                       framealpha=0.95, edgecolor='black', fancybox=True)
    legend.get_frame().set_linewidth(1.5)
    
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.tick_params(labelsize=12)
    
    # 右: 累積分布
    ax2 = axes[1]
    
    baseline_sorted = np.sort(baseline_abs_errors)
    pfp_sorted = np.sort(pfp_abs_errors)
    
    baseline_cumulative = np.arange(1, len(baseline_sorted) + 1) / len(baseline_sorted) * 100
    pfp_cumulative = np.arange(1, len(pfp_sorted) + 1) / len(pfp_sorted) * 100
    
    ax2.plot(baseline_sorted, baseline_cumulative, linewidth=3.5, 
             color='cornflowerblue', label=f'Baseline (MAE: {baseline_mae:.4f} D)', 
             alpha=0.8, zorder=2)
    ax2.plot(pfp_sorted, pfp_cumulative, linewidth=3.5, 
             color='mediumseagreen', label=f'EGNN×PFP (MAE: {pfp_mae:.4f} D)', 
             alpha=0.8, zorder=3)
    
    percentiles = [50, 75, 90, 95]
    colors_p = ['gray', 'gray', 'darkgray', 'darkgray']
    styles_p = ['-', '--', '-.', ':']
    
    for i, percentile in enumerate(percentiles):
        ax2.axhline(y=percentile, color=colors_p[i], linestyle=styles_p[i], 
                   linewidth=1.2, alpha=0.5, zorder=1)
        
        baseline_val = np.percentile(baseline_abs_errors, percentile)
        pfp_val = np.percentile(pfp_abs_errors, percentile)
        diff = baseline_val - pfp_val
        
        mid_x = (baseline_val + pfp_val) / 2
        ax2.plot([pfp_val, baseline_val], [percentile, percentile], 
                'k-', linewidth=2, alpha=0.4, zorder=4)
        ax2.text(mid_x, percentile - 3, f'Δ={diff:.4f}', 
                fontsize=9, ha='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='black', alpha=0.7))
    
    ax2.set_xlabel('Absolute Error (D)', fontsize=15, fontweight='bold')
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=15, fontweight='bold')
    ax2.set_title('Cumulative Distribution of Absolute Errors\n(PFP curve is left-shifted = better)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlim(0, xlim)
    ax2.set_ylim(0, 100)
    
    legend2 = ax2.legend(fontsize=12, loc='lower right', framealpha=0.9,
                        edgecolor='black', fancybox=True)
    legend2.get_frame().set_linewidth(1.5)
    
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.tick_params(labelsize=12)
    
    baseline_median = np.median(baseline_abs_errors)
    pfp_median = np.median(pfp_abs_errors)
    median_improvement = (baseline_median - pfp_median) / baseline_median * 100
    
    stats_box_text = (f'Key Statistics:\n'
                     f'Median Error\n'
                     f'  Baseline: {baseline_median:.4f} D\n'
                     f'  PFP: {pfp_median:.4f} D\n'
                     f'  ↓ {median_improvement:.1f}%\n\n'
                     f'90th Percentile\n'
                     f'  Baseline: {np.percentile(baseline_abs_errors, 90):.4f} D\n'
                     f'  PFP: {np.percentile(pfp_abs_errors, 90):.4f} D')
    
    ax2.text(0.02, 0.98, stats_box_text,
             transform=ax2.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85, 
                      edgecolor='navy', linewidth=1.5),
             fontweight='bold', family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_error_comparison_combined.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
        
    # ===================================================================
    # 2. ヒストグラム単体（横幅を狭く、凡例を適度なサイズに）
    # ===================================================================
    fig_hist = plt.figure(figsize=(8, 6))
    ax_hist = fig_hist.add_subplot(111)

    bins = np.linspace(-xlim, xlim, 100)

    ax_hist.hist(baseline_errors, bins=bins, alpha=0.5, 
                color='cornflowerblue', edgecolor='steelblue', 
                linewidth=1.0, label=f'Baseline (MAE: {baseline_mae:.4f} D)', zorder=2)

    ax_hist.hist(pfp_errors, bins=bins, alpha=0.5, 
                color='mediumseagreen', edgecolor='seagreen', 
                linewidth=1.0, label=f'EGNN×PFP (MAE: {pfp_mae:.4f} D)', zorder=1)

    ax_hist.axvline(x=0, color='red', linestyle='-', linewidth=2.5, 
                    zorder=10, alpha=0.9, label='Zero Error')

    ax_hist.axvline(x=baseline_errors.mean(), color='blue', 
                    linestyle='--', linewidth=1.5, alpha=0.5, zorder=5)
    ax_hist.axvline(x=pfp_errors.mean(), color='green', 
                    linestyle='--', linewidth=1.5, alpha=0.5, zorder=5)

    ax_hist.set_xlabel('Prediction Error (D)', fontsize=15, fontweight='bold')
    ax_hist.set_ylabel('Frequency', fontsize=15, fontweight='bold')
    ax_hist.set_xlim(-xlim, xlim)

    title_text_hist = (f'Error Distribution Comparison\n'
                    f'Baseline: MAE={baseline_mae:.4f} D, Std={baseline_std:.4f} D\n'
                    f'PFP: MAE={pfp_mae:.4f} D, Std={pfp_std:.4f} D\n'
                    f'Improvement: MAE ↓{mae_improvement:.1f}%, RMSE ↓{rmse_improvement:.1f}%')
    ax_hist.set_title(title_text_hist, fontsize=12, fontweight='bold', pad=15)

    # 凡例のフォントサイズを12に調整
    legend_hist = ax_hist.legend(fontsize=12, loc='upper right',  # 14 → 12
                                framealpha=0.95, edgecolor='black', fancybox=True)
    legend_hist.get_frame().set_linewidth(1.5)

    ax_hist.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax_hist.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_error_histogram_only.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===================================================================
    # 3. 累積分布単体
    # ===================================================================
    fig_cdf = plt.figure(figsize=(10, 6))
    ax_cdf = fig_cdf.add_subplot(111)
    
    ax_cdf.plot(baseline_sorted, baseline_cumulative, linewidth=3.5, 
                color='cornflowerblue', label=f'Baseline (MAE: {baseline_mae:.4f} D)', 
                alpha=0.8, zorder=2)
    ax_cdf.plot(pfp_sorted, pfp_cumulative, linewidth=3.5, 
                color='mediumseagreen', label=f'EGNN×PFP (MAE: {pfp_mae:.4f} D)', 
                alpha=0.8, zorder=3)
    
    for i, percentile in enumerate(percentiles):
        ax_cdf.axhline(y=percentile, color=colors_p[i], linestyle=styles_p[i], 
                       linewidth=1.2, alpha=0.5, zorder=1)
        
        baseline_val = np.percentile(baseline_abs_errors, percentile)
        pfp_val = np.percentile(pfp_abs_errors, percentile)
        diff = baseline_val - pfp_val
        
        mid_x = (baseline_val + pfp_val) / 2
        ax_cdf.plot([pfp_val, baseline_val], [percentile, percentile], 
                    'k-', linewidth=2, alpha=0.4, zorder=4)
        ax_cdf.text(mid_x, percentile - 3, f'Δ={diff:.4f}', 
                    fontsize=9, ha='center', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='black', alpha=0.7))
    
    ax_cdf.set_xlabel('Absolute Error (D)', fontsize=15, fontweight='bold')
    ax_cdf.set_ylabel('Cumulative Percentage (%)', fontsize=15, fontweight='bold')
    ax_cdf.set_title('Cumulative Distribution of Absolute Errors\n(PFP curve is left-shifted = better)', 
                     fontsize=12, fontweight='bold', pad=15)
    ax_cdf.set_xlim(0, xlim)
    ax_cdf.set_ylim(0, 100)
    
    legend_cdf = ax_cdf.legend(fontsize=12, loc='lower right', framealpha=0.9,
                               edgecolor='black', fancybox=True)
    legend_cdf.get_frame().set_linewidth(1.5)
    
    ax_cdf.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax_cdf.tick_params(labelsize=12)
    
    ax_cdf.text(0.02, 0.98, stats_box_text,
                transform=ax_cdf.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85, 
                         edgecolor='navy', linewidth=1.5),
                fontweight='bold', family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_error_cdf_only.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Plots saved:")
    print(f"  - Combined: test_error_comparison_combined.png")
    print(f"  - Histogram: test_error_histogram_only.png")
    print(f"  - CDF: test_error_cdf_only.png")
    
    # 統計テキスト出力
    stats_text = f"""
{'='*70}
QM9 B3LYP Dipole Moment Prediction - Test Set Error Analysis
{'='*70}

Model Comparison:
  - Baseline: EGNN with 5-dim one-hot features
  - PFP:      EGNN with 256-dim PFP + geometric features

{'='*70}
Error Statistics
{'='*70}

{'Metric':<25} {'Baseline':<15} {'PFP':<15} {'Improvement'}
{'-'*70}
{'MAE (D)':<25} {baseline_mae:<15.4f} {pfp_mae:<15.4f} {mae_improvement:>13.1f}%
{'RMSE (D)':<25} {baseline_rmse:<15.4f} {pfp_rmse:<15.4f} {rmse_improvement:>13.1f}%
{'Std Dev (D)':<25} {baseline_std:<15.4f} {pfp_std:<15.4f} {(baseline_std-pfp_std)/baseline_std*100:>13.1f}%
{'Mean Error (D)':<25} {baseline_errors.mean():<15.4f} {pfp_errors.mean():<15.4f} {'-':>14}
{'Median |Error| (D)':<25} {np.median(baseline_abs_errors):<15.4f} {np.median(pfp_abs_errors):<15.4f} {(np.median(baseline_abs_errors)-np.median(pfp_abs_errors))/np.median(baseline_abs_errors)*100:>13.1f}%

{'='*70}
Percentile Analysis (Absolute Error)
{'='*70}

{'Percentile':<15} {'Baseline (D)':<20} {'PFP (D)':<20} {'Difference (D)'}
{'-'*70}
"""
    
    for percentile in [50, 75, 90, 95, 99]:
        baseline_p = np.percentile(baseline_abs_errors, percentile)
        pfp_p = np.percentile(pfp_abs_errors, percentile)
        diff = baseline_p - pfp_p
        stats_text += f"{percentile}%{'':<12} {baseline_p:<20.4f} {pfp_p:<20.4f} {diff:>14.4f}\n"
    
    stats_text += f"\n{'='*70}\n"
    stats_text += f"Test Set Size: {len(baseline_errors):,} molecules\n"
    stats_text += f"{'='*70}\n"
    
    with open(os.path.join(output_dir, 'test_error_statistics.txt'), 'w') as f:
        f.write(stats_text)
    
    print(stats_text)

# ===================================================================
    # 4. y-y プロット（Inset付き・改善版）
    # ===================================================================
    # 必要なデータが渡されている場合のみ実行
    if all(v is not None for v in [baseline_targets, pfp_targets, baseline_preds, pfp_preds]):
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
        print("\n" + "="*70)
        print("📈 Generating Overlay y-y Plot with Inset")
        print("="*70)
        
        # --- 全体の軸範囲設定 ---
        all_targets_concat = np.concatenate([baseline_targets, pfp_targets])
        all_preds_concat = np.concatenate([baseline_preds, pfp_preds])
        min_val = min(all_targets_concat.min(), all_preds_concat.min())
        max_val = max(all_targets_concat.max(), all_preds_concat.max())
        
        # 軸範囲に余裕を持たせる
        range_val = max_val - min_val
        margin = range_val * 0.02
        plot_min = min_val - margin
        plot_max = max_val + margin
        
        fig_yy = plt.figure(figsize=(12, 11))
        ax_yy = fig_yy.add_subplot(111)
        
        # --- メインプロット（全体） ---
        # Baseline（下層・やや薄く）
        ax_yy.scatter(baseline_targets, baseline_preds, alpha=0.5, s=25, 
                      edgecolors='none', c='cornflowerblue', 
                      label=f'Baseline (MAE={baseline_mae:.3f}D)', zorder=2)
        # PFP（上層・濃く）
        ax_yy.scatter(pfp_targets, pfp_preds, alpha=0.6, s=25, 
                      edgecolors='none', c='mediumseagreen', 
                      label=f'EGNN-PFP (MAE={pfp_mae:.3f}D)', zorder=3)
        
        # 対角線
        ax_yy.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=2.5, zorder=1)
        
        ax_yy.set_xlabel('True Dipole Moment (D)', fontsize=38, fontweight='bold')
        ax_yy.set_ylabel('Predicted Dipole Moment (D)', fontsize=38, fontweight='bold')
        ax_yy.set_xlim(plot_min, plot_max)
        ax_yy.set_ylim(plot_min, plot_max)
        ax_yy.set_aspect('equal')
        ax_yy.tick_params(axis='both', which='major', labelsize=24)
        ax_yy.grid(True, alpha=0.3, linewidth=1.2)
        
        # 凡例を左上に配置（インセットが右下なので干渉しない）
        legend_yy = ax_yy.legend(loc='upper left', fontsize=28, framealpha=0.95,
                                 edgecolor='black', fancybox=False, shadow=False,
                                 markerscale=2.5)
        legend_yy.get_frame().set_linewidth(1.5)
        
        # 枠線を追加
        for spine in ax_yy.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        
        # --- インセット（拡大図）の作成 ---
        # 右下に配置、サイズを50%に拡大
        axins = inset_axes(ax_yy, width="50%", height="50%", 
                          loc='lower right', borderpad=2.5)
        
        # 拡大図にもプロット（マーカーサイズを調整）
        axins.scatter(baseline_targets, baseline_preds, alpha=0.6, s=30, 
                      edgecolors='none', c='cornflowerblue', zorder=2)
        axins.scatter(pfp_targets, pfp_preds, alpha=0.7, s=30, 
                      edgecolors='none', c='mediumseagreen', zorder=3)
        
        # 対角線（拡大図用）
        axins.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=2.0, zorder=1)
        
        # ★★★ 拡大範囲の自動決定 ★★★
        # データの中央値周辺を拡大（最も密集している領域）
        median_val = np.median(all_targets_concat)
        # 中央値±3Dの範囲を拡大（調整可能）
        zoom_range = 3.0
        x1, x2 = max(0, median_val - zoom_range), median_val + zoom_range
        y1, y2 = max(0, median_val - zoom_range), median_val + zoom_range
        
        # または、特定の範囲を指定する場合:
        # x1, x2 = 0, 8  # 0-8 Debye
        # y1, y2 = 0, 8
        
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_aspect('equal')
        
        # 拡大図の目盛りとグリッド
        axins.tick_params(labelsize=16, width=1.5)
        axins.grid(True, alpha=0.4, linewidth=1.0)
        
        # 拡大図の枠線を強調
        for spine in axins.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2.0)
        
        # メイン図と拡大図をつなぐ線を描画（赤色で強調）
        mark_inset(ax_yy, axins, loc1=2, loc2=4, fc="none", ec="red", lw=2.0)
        
        # 拡大領域を示す矩形をメイン図に描画
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        fill=False, edgecolor='red', linewidth=2.0, 
                        linestyle='--', zorder=4)
        ax_yy.add_patch(rect)
        
        plt.tight_layout(pad=1.5)
        plt.savefig(os.path.join(output_dir, 'test_baseline_vs_pfp_overlay_dipole_inset.png'), 
                    dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"✓ Overlay y-y plot with inset saved: test_baseline_vs_pfp_overlay_dipole_inset.png")
        print(f"  Zoom region: [{x1:.1f}, {x2:.1f}] D")
# =============================================
# メイン関数
# =============================================

def main():
    QM9_DIPOLE_DIR = "/home/users/uchiyama/QM9_dipole"
    
    # モデルパス
    BASELINE_MODEL_PATH = os.path.join(QM9_DIPOLE_DIR, 
        "training_baseline_dipole_B3LYP_0115/best_baseline_model.pth")
    PFP_MODEL_PATH = os.path.join(QM9_DIPOLE_DIR, 
        "training_pfp_B3LYP_egnn_official_128_attention_perfect_0115/best_pfp_official_model.pth")
    
    # データパス
    BASELINE_GRAPH_DIR = os.path.join(QM9_DIPOLE_DIR, "graphs_dipole_qm9_B3LYP_baseline_perfect")
    PFP_GRAPH_DIR = os.path.join(QM9_DIPOLE_DIR, "graphs_dipole_qm9_B3LYP_pfp_perfect")
    
    OUTPUT_DIR = os.path.join(QM9_DIPOLE_DIR, "test_error_analysis")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("QM9 B3LYP 双極子予測 - テスト誤差ヒストグラム比較")
    print("="*70)
    print(f"Device: {device}\n")
    
    # テストデータ読み込み
    print("🔹 Loading test data...")
    baseline_test = torch.load(os.path.join(BASELINE_GRAPH_DIR, "test_graphs.pt"), 
                               weights_only=False)
    pfp_test = torch.load(os.path.join(PFP_GRAPH_DIR, "test_graphs.pt"), 
                          weights_only=False)
    
    print(f"✓ Baseline test: {len(baseline_test):,} molecules")
    print(f"✓ PFP test:      {len(pfp_test):,} molecules\n")
    
    # DataLoader作成
    batch_size = 96
    baseline_loader = DataLoader(baseline_test, batch_size=batch_size, 
                                 shuffle=False, num_workers=4)
    pfp_loader = DataLoader(pfp_test, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    # Baselineモデル読み込み
    print("🔹 Loading Baseline model...")
    baseline_model = EGNN_Official(
        in_node_nf=5,
        in_edge_nf=0,
        hidden_nf=128,
        device=device,
        n_layers=7,
        attention=True,
        node_attr=True
    ).to(device)
    
    baseline_checkpoint = torch.load(BASELINE_MODEL_PATH, map_location=device, weights_only=False)
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    print(f"✓ Loaded from epoch {baseline_checkpoint['epoch']}")
    print(f"✓ Val MAE: {baseline_checkpoint['val_mae']:.4f} D\n")
    
    # PFPモデル読み込み
    print("🔹 Loading PFP model...")
    pfp_model = EGNN_Official(
        in_node_nf=261,
        in_edge_nf=4,
        hidden_nf=128,
        device=device,
        n_layers=7,
        attention=True,
        node_attr=True
    ).to(device)
    
    pfp_checkpoint = torch.load(PFP_MODEL_PATH, map_location=device, weights_only=False)
    pfp_model.load_state_dict(pfp_checkpoint['model_state_dict'])
    print(f"✓ Loaded from epoch {pfp_checkpoint['epoch']}")
    print(f"✓ Val MAE: {pfp_checkpoint['val_mae']:.4f} D\n")
    
    # テストセット評価
    print("🔹 Evaluating Baseline model on test set...")
    baseline_preds, baseline_targets = evaluate_model(
        baseline_model, baseline_loader, device, is_pfp=False)
    baseline_errors = baseline_preds - baseline_targets
    
    print("🔹 Evaluating PFP model on test set...")
    pfp_preds, pfp_targets = evaluate_model(
        pfp_model, pfp_loader, device, is_pfp=True)
    pfp_errors = pfp_preds - pfp_targets
    
    # ヒストグラムプロット（修正版）
    print("\n🔹 Generating error histograms and y-y plot...")
    # グローバル変数として targets も渡す
    plot_error_histograms_improved.__globals__['baseline_targets'] = baseline_targets
    plot_error_histograms_improved.__globals__['pfp_targets'] = pfp_targets
    plot_error_histograms_improved.__globals__['baseline_preds'] = baseline_preds
    plot_error_histograms_improved.__globals__['pfp_preds'] = pfp_preds
    plot_error_histograms_improved(baseline_errors, pfp_errors, OUTPUT_DIR) 
    
    print(f"\n✅ Analysis complete! Results saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
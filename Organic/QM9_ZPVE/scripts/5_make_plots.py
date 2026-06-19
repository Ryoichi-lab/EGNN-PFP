#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QM9 B3LYP ZPVE (Zero-point vibrational energy) テスト - EGNN×PFP
訓練済みモデルの評価とテストセットでの性能測定
単位: meV
"""

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns

# =============================================
# Utility functions
# =============================================

def unsorted_segment_sum(data, segment_ids, num_segments):
    """セグメント単位の和"""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    """セグメント単位の平均"""
    result_shape = (num_segments, data.size(1))
    segment_ids_expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids_expanded, data)
    count.scatter_add_(0, segment_ids_expanded, torch.ones_like(data))
    return result / count.clamp(min=1)


# =============================================
# E_GCL_mask Layer (EGNN公式実装)
# =============================================

class E_GCL_mask(nn.Module):
    """E_GCL with masking support (EGNN公式実装)"""
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
# EGNN公式実装
# =============================================

class EGNN_Official(nn.Module):
    """EGNN公式実装の忠実な再現"""
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
# データ前処理
# =============================================

def prepare_batch_official(batch, device):
    """PyG形式のバッチを公式EGNN形式に変換"""
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
    
    offset = 0
    for i, n_nodes in enumerate(node_counts):
        mask = (batch.batch == i)
        start_idx = i * max_nodes
        
        h0[start_idx:start_idx + n_nodes] = batch.x[mask].to(device)
        x[start_idx:start_idx + n_nodes] = batch.pos[mask].to(device)
        node_mask[start_idx:start_idx + n_nodes] = 1.0
        
        offset += n_nodes
    
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


# =============================================
# テスト関数
# =============================================

def test_model(model, loader, device, mean, mad, property_name="ZPVE"):
    """テストセットでモデルを評価"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    HA_TO_MEV = 27211.4
    
    print(f"\n🔹 Testing on {len(loader.dataset)} molecules...")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            batch = batch.to(device)
            
            h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, target = \
                prepare_batch_official(batch, device)
            
            pred = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
            
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"\n⚠️  Warning: NaN/Inf detected in predictions!")
                continue
            
            pred_original = pred * mad + mean
            
            all_preds.append((pred_original * HA_TO_MEV).cpu().numpy())
            all_targets.append((target * HA_TO_MEV).cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # メトリクス計算
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    
    # 相対誤差
    relative_errors = np.abs(all_preds - all_targets) / (np.abs(all_targets) + 1e-8)
    mean_relative_error = np.mean(relative_errors) * 100
    
    # パーセンタイル誤差
    errors = np.abs(all_preds - all_targets)
    percentiles = {
        '50th': np.percentile(errors, 50),
        '90th': np.percentile(errors, 90),
        '95th': np.percentile(errors, 95),
        '99th': np.percentile(errors, 99)
    }
    
    return {
        'predictions': all_preds,
        'targets': all_targets,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mean_relative_error': mean_relative_error,
        'percentiles': percentiles,
        'n_samples': len(all_preds)
    }


# =============================================
# 可視化関数
# =============================================

def plot_comprehensive_results(results, output_dir, property_name="ZPVE", paper_mae=None):
    """包括的な結果の可視化"""
    
    predictions = results['predictions']
    targets = results['targets']
    mae = results['mae']
    rmse = results['rmse']
    r2 = results['r2']
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Parity Plot (大きめ)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    # ヘキサビンでの密度プロット
    hb = ax1.hexbin(targets, predictions, gridsize=50, cmap='YlOrRd', mincnt=1)
    
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'b--', lw=2, label='Perfect prediction')
    
    ax1.set_xlabel(f'True {property_name} (meV)', fontsize=14, fontweight='bold')
    ax1.set_ylabel(f'Predicted {property_name} (meV)', fontsize=14, fontweight='bold')
    ax1.set_title(f'EGNN×PFP Test Set Performance\nMAE: {mae:.4f} meV | RMSE: {rmse:.4f} meV | R²: {r2:.8f}', 
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(hb, ax=ax1)
    cbar.set_label('Count', fontsize=12)
    
    # 2. 誤差分布
    ax2 = fig.add_subplot(gs[0, 2])
    errors = predictions - targets
    ax2.hist(errors, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(errors):.4f}')
    ax2.set_xlabel('Prediction Error (meV)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Error Distribution\nStd: {np.std(errors):.4f} meV', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. 絶対誤差分布
    ax3 = fig.add_subplot(gs[1, 2])
    abs_errors = np.abs(errors)
    ax3.hist(abs_errors, bins=100, alpha=0.7, edgecolor='black', color='coral')
    ax3.axvline(x=mae, color='red', linestyle='--', linewidth=2, label=f'MAE: {mae:.4f}')
    if paper_mae is not None:
        ax3.axvline(x=paper_mae, color='blue', linestyle='--', linewidth=2, 
                   label=f'Paper: {paper_mae:.4f}')
    ax3.set_xlabel('Absolute Error (meV)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 相対誤差 vs 真値
    ax4 = fig.add_subplot(gs[2, 0])
    relative_errors = np.abs(errors) / (np.abs(targets) + 1e-8) * 100
    ax4.scatter(targets, relative_errors, alpha=0.3, s=10)
    ax4.axhline(y=np.median(relative_errors), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(relative_errors):.2f}%')
    ax4.set_xlabel(f'True {property_name} (meV)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Relative Error vs True Value', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. 累積分布
    ax5 = fig.add_subplot(gs[2, 1])
    sorted_abs_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors) * 100
    ax5.plot(sorted_abs_errors, cumulative, linewidth=2, color='darkgreen')
    ax5.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax5.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90%')
    ax5.axhline(y=95, color='purple', linestyle='--', alpha=0.5, label='95%')
    ax5.set_xlabel('Absolute Error (meV)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. 統計サマリー
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    stats_text = f"""
    Test Set Statistics
    {'='*40}
    
    Samples: {results['n_samples']:,}
    
    Performance Metrics:
    • MAE:  {mae:.6f} meV
    • RMSE: {rmse:.6f} meV
    • R²:   {r2:.10f}
    
    Relative Error:
    • Mean: {results['mean_relative_error']:.4f}%
    
    Error Percentiles:
    • 50th: {results['percentiles']['50th']:.6f} meV
    • 90th: {results['percentiles']['90th']:.6f} meV
    • 95th: {results['percentiles']['95th']:.6f} meV
    • 99th: {results['percentiles']['99th']:.6f} meV
    
    Value Range:
    • Min:  {targets.min():.4f} meV
    • Max:  {targets.max():.4f} meV
    • Mean: {targets.mean():.4f} meV
    • Std:  {targets.std():.4f} meV
    """
    
    if paper_mae is not None:
        improvement = (paper_mae - mae) / paper_mae * 100
        stats_text += f"\n    vs EGNN Paper:\n    • Improvement: {improvement:+.2f}%"
    
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(os.path.join(output_dir, f'{property_name}_test_comprehensive_results.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comprehensive plot saved")


# =============================================
# メイン関数
# =============================================

def main():
    # パス設定
    QM9_ZPVE_DIR = "/home/users/uchiyama/QM9_ZPVE"
    GRAPH_DIR = "/home/users/uchiyama/QM9_ZPVE/graphs_zpve_qm9_B3LYP_pfp"
    MODEL_DIR = "/home/users/uchiyama/QM9_ZPVE/QM9_ZPVE_training_pfp_egnn_1219"
    OUTPUT_DIR = os.path.join(MODEL_DIR, "test_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("QM9 B3LYP ZPVE Test - EGNN×PFP")
    print("="*70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # テストデータ読み込み
    print("\n🔹 Loading test data...")
    test_graphs = torch.load(os.path.join(GRAPH_DIR, "test_graphs.pt"), weights_only=False)
    print(f"✓ Test: {len(test_graphs):,} molecules")
    
    # チェックポイント読み込み
    print("\n🔹 Loading best model checkpoint...")
    checkpoint_path = os.path.join(MODEL_DIR, "best_pfp_official_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    mean = torch.tensor(checkpoint['mean'], dtype=torch.float32, device=device)
    mad = torch.tensor(checkpoint['mad'], dtype=torch.float32, device=device)
    
    print(f"✓ Checkpoint from epoch {checkpoint['epoch']}")
    print(f"✓ Val MAE: {checkpoint['val_mae']:.4f} meV")
    print(f"✓ Normalization: mean={mean.item():.6f} Ha, MAD={mad.item():.6f} Ha")
    
    # モデル初期化
    print("\n🔹 Initializing model...")
    model = EGNN_Official(
        in_node_nf=261,
        in_edge_nf=4,
        hidden_nf=128,
        device=device,
        act_fn=nn.SiLU(),
        n_layers=7,
        coords_weight=1.0,
        attention=False,
        node_attr=True
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded successfully")
    
    # テストローダー作成
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False, num_workers=0)
    
    # テスト実行
    print("\n" + "="*70)
    print("🚀 Starting Test Evaluation")
    print("="*70)
    
    results = test_model(model, test_loader, device, mean, mad, property_name="ZPVE")
    
    # 結果表示
    print("\n" + "="*70)
    print("📊 Test Results")
    print("="*70)
    print(f"Samples:              {results['n_samples']:,}")
    print(f"MAE:                  {results['mae']:.6f} meV")
    print(f"RMSE:                 {results['rmse']:.6f} meV")
    print(f"R²:                   {results['r2']:.10f}")
    print(f"Mean Relative Error:  {results['mean_relative_error']:.4f}%")
    print(f"\nError Percentiles:")
    print(f"  50th percentile:    {results['percentiles']['50th']:.6f} meV")
    print(f"  90th percentile:    {results['percentiles']['90th']:.6f} meV")
    print(f"  95th percentile:    {results['percentiles']['95th']:.6f} meV")
    print(f"  99th percentile:    {results['percentiles']['99th']:.6f} meV")
    
    # EGNN論文との比較
    PAPER_MAE_MEV = 1.5
    improvement = (PAPER_MAE_MEV - results['mae']) / PAPER_MAE_MEV * 100
    print(f"\nComparison with EGNN Paper:")
    print(f"  Paper MAE:          {PAPER_MAE_MEV:.4f} meV")
    print(f"  Our MAE:            {results['mae']:.4f} meV")
    print(f"  Improvement:        {improvement:+.2f}%")
    print("="*70)
    
    # 可視化
    print("\n🔹 Creating visualizations...")
    plot_comprehensive_results(results, OUTPUT_DIR, property_name="ZPVE", paper_mae=PAPER_MAE_MEV)
    
    # 結果をJSON形式で保存
    test_stats = {
        'dataset': 'QM9',
        'property': 'ZPVE',
        'test_set_size': results['n_samples'],
        'metrics': {
            'mae_meV': float(results['mae']),
            'rmse_meV': float(results['rmse']),
            'r2': float(results['r2']),
            'mean_relative_error_percent': float(results['mean_relative_error'])
        },
        'error_percentiles_meV': {
            '50th': float(results['percentiles']['50th']),
            '90th': float(results['percentiles']['90th']),
            '95th': float(results['percentiles']['95th']),
            '99th': float(results['percentiles']['99th'])
        },
        'comparison': {
            'paper_egnn_mae_meV': PAPER_MAE_MEV,
            'improvement_percent': float(improvement)
        },
        'model_checkpoint': {
            'epoch': checkpoint['epoch'],
            'val_mae_meV': float(checkpoint['val_mae'])
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(OUTPUT_DIR, 'test_results_zpve.json'), 'w') as f:
        json.dump(test_stats, f, indent=2)
    
    print(f"✓ Test results saved to: {OUTPUT_DIR}")
    
    print("\n" + "="*70)
    print("🎉 Test Evaluation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
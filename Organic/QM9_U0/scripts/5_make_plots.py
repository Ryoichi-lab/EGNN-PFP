#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QM9 B3LYP u (Internal energy at 0K) テスト評価スクリプト
EGNN×PFP公式実装モデルの性能評価
"""

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import os
import json
from datetime import datetime
from tqdm import tqdm
import math

# =============================================
# Utility functions (訓練スクリプトから複製)
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
# E_GCL_mask Layer (訓練スクリプトから複製)
# =============================================

class E_GCL_mask(nn.Module):
    """
    E_GCL with masking support (EGNN公式実装)
    座標更新を無効化したバージョン
    """
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
        
        # Edge model
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )
        
        # Node model
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_attr_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )
        
        # Attention
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )
        
        self.act_fn = act_fn
    
    def edge_model(self, source, target, radial, edge_attr):
        """エッジ特徴の計算"""
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
        """ノード特徴の更新"""
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
        """座標から距離を計算"""
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, dim=1, keepdim=True)
        
        norm = torch.sqrt(radial).detach() + self.epsilon
        coord_diff = coord_diff / norm
        
        return radial, coord_diff
    
    def forward(self, h, edge_index, coord, node_mask, edge_mask, 
                edge_attr=None, node_attr=None, n_nodes=None):
        """順伝播（座標更新なし）"""
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        edge_feat = edge_feat * edge_mask
        
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        
        return h, coord, edge_attr


# =============================================
# EGNN公式実装 (訓練スクリプトから複製)
# =============================================

class EGNN_Official(nn.Module):
    """
    EGNN公式実装の忠実な再現
    QM9 u予測用
    """
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', 
                 act_fn=nn.SiLU(), n_layers=7, coords_weight=1.0, 
                 attention=False, node_attr=True):
        super(EGNN_Official, self).__init__()
        
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.node_attr = node_attr
        
        # Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        
        # EGNN layers
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
        
        # Decoders
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
        """順伝播"""
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
# データ前処理 (訓練スクリプトから複製)
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
# テスト評価関数
# =============================================

def evaluate_test_set(model, test_loader, device, mean, mad, output_dir):
    """テストセットで詳細評価"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_errors = []
    
    HA_TO_MEV = 27211.4
    
    print("🔍 テストセット評価中...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = batch.to(device)
            
            h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, target = \
                prepare_batch_official(batch, device)
            
            pred = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
            
            # 元のスケールに戻す
            pred_original = pred * mad + mean
            
            # meV単位に変換
            pred_mev = (pred_original * HA_TO_MEV).cpu().numpy()
            target_mev = (target * HA_TO_MEV).cpu().numpy()
            
            all_preds.append(pred_mev)
            all_targets.append(target_mev)
            all_errors.append(pred_mev - target_mev)
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_errors = np.concatenate(all_errors)
    
    # 統計量計算
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    
    # パーセンタイル
    percentiles = [50, 90, 95, 99]
    abs_errors = np.abs(all_errors)
    percentile_values = {p: np.percentile(abs_errors, p) for p in percentiles}
    
    # 相関係数
    pearson_r, pearson_p = stats.pearsonr(all_targets, all_preds)
    spearman_r, spearman_p = stats.spearmanr(all_targets, all_preds)
    
    results = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'mean_error': float(all_errors.mean()),
        'std_error': float(all_errors.std()),
        'median_error': float(np.median(all_errors)),
        'percentiles': {f'p{p}': float(v) for p, v in percentile_values.items()},
        'num_samples': len(all_targets),
        'timestamp': datetime.now().isoformat()
    }
    
    return results, all_preds, all_targets, all_errors


def plot_comprehensive_analysis(preds, targets, errors, results, output_dir):
    """包括的な分析プロット"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Parity plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(targets, preds, alpha=0.3, s=15, edgecolors='none', c='blue')
    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('True u (meV)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted u (meV)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Parity Plot\nMAE: {results["mae"]:.2f} meV, R²: {results["r2"]:.4f}', 
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(errors, bins=100, alpha=0.7, edgecolor='black', color='green')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax2.axvline(x=results['mean_error'], color='blue', linestyle='--', 
                linewidth=2, label=f'Mean: {results["mean_error"]:.2f}')
    ax2.set_xlabel('Prediction Error (meV)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Error Distribution\nStd: {results["std_error"]:.2f} meV', 
                  fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residual plot
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(targets, errors, alpha=0.3, s=15, edgecolors='none', c='purple')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax3.axhline(y=results['mean_error'], color='blue', linestyle='--', linewidth=2)
    ax3.fill_between([targets.min(), targets.max()], 
                     -results['std_error'], results['std_error'], 
                     alpha=0.2, color='gray', label='±1 std')
    ax3.set_xlabel('True u (meV)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Prediction Error (meV)', fontsize=12, fontweight='bold')
    ax3.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Absolute error vs true value
    ax4 = plt.subplot(2, 3, 4)
    abs_errors = np.abs(errors)
    ax4.scatter(targets, abs_errors, alpha=0.3, s=15, edgecolors='none', c='orange')
    ax4.axhline(y=results['mae'], color='red', linestyle='--', 
                linewidth=2, label=f'MAE: {results["mae"]:.2f}')
    ax4.set_xlabel('True u (meV)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Absolute Error (meV)', fontsize=12, fontweight='bold')
    ax4.set_title('Absolute Error Distribution', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 5. Q-Q plot
    ax5 = plt.subplot(2, 3, 5)
    stats.probplot(errors, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot (Normality Check)', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative error distribution
    ax6 = plt.subplot(2, 3, 6)
    sorted_abs_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors) * 100
    ax6.plot(sorted_abs_errors, cumulative, linewidth=2, color='darkblue')
    ax6.axvline(x=results['mae'], color='red', linestyle='--', 
                linewidth=2, label=f'MAE: {results["mae"]:.2f}')
    for p in [50, 90, 95, 99]:
        val = results['percentiles'][f'p{p}']
        ax6.axvline(x=val, color='gray', linestyle=':', alpha=0.7,
                   label=f'P{p}: {val:.2f}')
    ax6.set_xlabel('Absolute Error (meV)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax6.set_title('Cumulative Error Distribution', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_comprehensive_analysis.png'), dpi=300)
    plt.close()
    
    print("✓ 包括的分析プロット保存完了")


def print_detailed_results(results):
    """詳細な結果を表示"""
    
    PAPER_MAE_MEV = 0.014 * 27211.4  # EGNN論文のu MAE
    
    print("\n" + "="*70)
    print("📊 テスト評価結果")
    print("="*70)
    print(f"サンプル数: {results['num_samples']:,}")
    print()
    
    print("【主要指標】")
    print(f"  MAE:          {results['mae']:.2f} meV")
    print(f"  RMSE:         {results['rmse']:.2f} meV")
    print(f"  R²:           {results['r2']:.4f}")
    print()
    
    print("【誤差統計】")
    print(f"  Mean Error:   {results['mean_error']:+.2f} meV")
    print(f"  Std Error:    {results['std_error']:.2f} meV")
    print(f"  Median Error: {results['median_error']:+.2f} meV")
    print()
    
    print("【パーセンタイル (絶対誤差)】")
    for p in [50, 90, 95, 99]:
        print(f"  P{p}:          {results['percentiles'][f'p{p}']:.2f} meV")
    print()
    
    print("【相関係数】")
    print(f"  Pearson r:    {results['pearson_r']:.4f} (p={results['pearson_p']:.2e})")
    print(f"  Spearman ρ:   {results['spearman_r']:.4f} (p={results['spearman_p']:.2e})")
    print()
    
    print("【ベンチマーク比較】")
    print(f"  EGNN論文 (u): {PAPER_MAE_MEV:.2f} meV")
    print(f"  本モデル:     {results['mae']:.2f} meV")
    if results['mae'] < PAPER_MAE_MEV:
        improvement = (1 - results['mae'] / PAPER_MAE_MEV) * 100
        print(f"  → {improvement:.1f}% 改善 ✓")
    else:
        degradation = (results['mae'] / PAPER_MAE_MEV - 1) * 100
        print(f"  → {degradation:.1f}% 劣化")
    
    print("="*70)


def main():
    # パス設定
    QM9_DIR = "/home/users/uchiyama/QM9_R2"
    GRAPH_DIR = "/home/users/uchiyama/QM9_R2/graphs_r2_qm9_B3LYP_pfp_perfect"
    MODEL_DIR = "/home/users/uchiyama/QM9_R2/QM9_R2_training_pfp_egnn_128_attention_L1_0109_perfect/checkpoint_epoch_1000.pth"
    OUTPUT_DIR = ("/home/users/uchiyama/QM9_R2/test")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("QM9 B3LYP u テスト評価")
    print("EGNN×PFP公式実装モデル")
    print("="*70)
    print(f"Device: {device}")
    
    # テストデータ読み込み
    print("\n🔹 テストデータ読み込み中...")
    test_graphs = torch.load(os.path.join(GRAPH_DIR, "test_graphs.pt"), 
                             weights_only=False)
    print(f"✓ Test: {len(test_graphs):,} molecules")
    
    # モデル読み込み
    print("\n🔹 訓練済みモデル読み込み中...")
    checkpoint_path = os.path.join(MODEL_DIR, 'checkpoint_epoch_1000.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    mean = torch.tensor(checkpoint['mean'], dtype=torch.float32, device=device)
    mad = torch.tensor(checkpoint['mad'], dtype=torch.float32, device=device)
    
    print(f"✓ Checkpoint loaded (Epoch {checkpoint['epoch']})")
    print(f"  Train MAE: {checkpoint['train_mae']:.2f} meV")
    print(f"  Val MAE:   {checkpoint['val_mae']:.2f} meV")
    
    # モデル初期化
    model = EGNN_Official(
        in_node_nf=261,
        in_edge_nf=4,
        hidden_nf=128,
        device=device,
        act_fn=nn.SiLU(),
        n_layers=7,
        coords_weight=1.0,
        attention=True,
        node_attr=True
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ モデルパラメータ復元完了")
    
    # データローダー作成
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False, num_workers=0)
    
    # 評価実行
    results, preds, targets, errors = evaluate_test_set(
        model, test_loader, device, mean, mad, OUTPUT_DIR
    )
    
    # 結果表示
    print_detailed_results(results)
    
    # 結果保存
    results_path = os.path.join(OUTPUT_DIR, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 結果保存: {results_path}")
    
    # 予測値保存
    np.savez(
        os.path.join(OUTPUT_DIR, 'test_predictions.npz'),
        predictions=preds,
        targets=targets,
        errors=errors
    )
    print(f"💾 予測値保存: test_predictions.npz")
    
    # プロット作成
    print("\n🎨 可視化作成中...")
    plot_comprehensive_analysis(preds, targets, errors, results, OUTPUT_DIR)
    
    print("\n✅ テスト評価完了!")
    print(f"📁 出力ディレクトリ: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QM9 B3LYP H (Internal energy at 0K) 予測 - EGNN Baseline
EGNN公式実装の忠実な再現（PFPなし）+ ターゲット標準化
PFP版グラフからone-hot特徴のみを抽出して使用
単位: meV
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
import math

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
# EGNN公式実装（Baseline）
# =============================================

class EGNN_Official(nn.Module):
    """
    EGNN公式実装の忠実な再現（Baseline: PFPなし）
    QM9 H予測用
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
        """
        Args:
            h0: ノード特徴 [total_nodes, in_node_nf]
            x: 座標 [total_nodes, 3]
            edges: エッジインデックス [2, num_edges]
            edge_attr: エッジ特徴 [num_edges, in_edge_nf]
            node_mask: ノードマスク [total_nodes, 1]
            edge_mask: エッジマスク [num_edges, 1]
            n_nodes: グラフあたりのノード数
        """
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
# データ前処理（公式実装形式に変換・Baseline用）
# =============================================

def create_atom_onehot(batch, device):
    """
    原子番号からone-hot特徴を作成（H, C, N, O, F）
    PFP特徴が含まれている場合は、原子番号部分のみを抽出
    """
    node_features = batch.x
    
    # ノード特徴の次元数で判定
    if node_features.size(1) == 5:
        # すでにone-hotの場合（ベースライン用グラフ）
        return node_features.to(device)
    
    elif node_features.size(1) == 261:
        # PFP版グラフ（256 PFP + 1 atomic_num + 4 geometric）
        # 257番目（index 256）が原子番号
        atomic_nums = node_features[:, 256].long()
        
        # one-hot特徴を作成
        features = torch.zeros(atomic_nums.size(0), 5, device=device)
        
        # H=1, C=6, N=7, O=8, F=9 -> one-hot index
        atom_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        for i, z in enumerate(atomic_nums):
            z_val = z.item()
            if z_val in atom_map:
                features[i, atom_map[z_val]] = 1
            else:
                # 未知の原子番号はHとして扱う
                features[i, 0] = 1
        
        return features
    
    else:
        raise ValueError(f"Unexpected node feature dimension: {node_features.size(1)}")


def prepare_batch_official_baseline(batch, device):
    """
    PyG形式のバッチを公式EGNN形式に変換（ベースライン用）
    PFP版グラフからもone-hot特徴のみを抽出可能
    """
    batch_size = batch.batch.max().item() + 1
    max_nodes = 0
    
    node_counts = []
    for i in range(batch_size):
        mask = (batch.batch == i)
        node_counts.append(mask.sum().item())
        max_nodes = max(max_nodes, mask.sum().item())
    
    total_nodes = batch_size * max_nodes
    
    # ★one-hot特徴を作成（PFP版からも抽出可能）
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
    
    # ★ベースラインではエッジ特徴なし（距離は座標から計算される）
    edge_attr = None
    
    return h0, x, edges, edge_attr, node_mask, edge_mask, max_nodes, batch.y.to(device)


# =============================================
# Cosine Annealing Scheduler
# =============================================

class CosineAnnealingScheduler:
    """Cosine Annealing学習率スケジューラ"""
    def __init__(self, optimizer, max_epochs, eta_min=1e-7):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.epoch = 0
    
    def step(self):
        progress = self.epoch / self.max_epochs
        lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.epoch += 1
        return lr
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


# =============================================
# ターゲット標準化用の関数
# =============================================

def compute_mean_mad(loader, device):
    """訓練データからmeanとMADを計算（EGNN原著準拠）"""
    all_targets = []
    
    print("  Collecting H target values...")
    for batch in loader:
        all_targets.append(batch.y.numpy())
    
    all_targets = np.concatenate(all_targets)
    mean = all_targets.mean()
    mad = np.abs(all_targets - mean).mean()
    
    HA_TO_MEV = 27211.4
    
    print(f"  H Target statistics:")
    print(f"    Mean: {mean:.6f} Ha ({mean * HA_TO_MEV:.2f} meV)")
    print(f"    MAD:  {mad:.6f} Ha ({mad * HA_TO_MEV:.2f} meV)")
    print(f"    Min:  {all_targets.min():.6f} Ha ({all_targets.min() * HA_TO_MEV:.2f} meV)")
    print(f"    Max:  {all_targets.max():.6f} Ha ({all_targets.max() * HA_TO_MEV:.2f} meV)")
    
    return mean, mad


# =============================================
# 訓練・評価関数（公式実装形式）
# =============================================

def train_epoch_official(model, loader, optimizer, device, mean, mad):
    """公式実装形式での訓練"""
    model.train()
    total_loss = 0
    total_mae = 0
    num_samples = 0
    
    HA_TO_MEV = 27211.4
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        
        h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, target = \
            prepare_batch_official_baseline(batch, device)
        
        optimizer.zero_grad()
        pred = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
        
        target_normalized = (target - mean) / mad
        loss = nn.MSELoss()(pred, target_normalized)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  Warning: NaN/Inf loss detected, skipping batch")
            continue
        
        pred_original = pred * mad + mean
        mae_ha = torch.abs(pred_original - target).mean()
        mae_mev = mae_ha * HA_TO_MEV
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if not torch.isnan(mae_mev) and not torch.isinf(mae_mev):
            total_loss += loss.item() * len(target)
            total_mae += mae_mev.item() * len(target)
            num_samples += len(target)
    
    if num_samples == 0:
        return float('inf'), float('inf')
    
    return total_loss / num_samples, total_mae / num_samples


def validate_epoch_official(model, loader, device, mean, mad):
    """公式実装形式での検証"""
    model.eval()
    total_mae = 0
    num_samples = 0
    
    all_preds = []
    all_targets = []
    
    HA_TO_MEV = 27211.4
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            batch = batch.to(device)
            
            h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, target = \
                prepare_batch_official_baseline(batch, device)
            
            pred = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
            
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"\n⚠️  Warning: NaN/Inf detected in predictions!")
                continue
            
            pred_original = pred * mad + mean
            mae_ha = torch.abs(pred_original - target).mean()
            mae_mev = mae_ha * HA_TO_MEV
            
            if not torch.isnan(mae_mev) and not torch.isinf(mae_mev):
                total_mae += mae_mev.item() * len(target)
                num_samples += len(target)
                
                all_preds.append((pred_original * HA_TO_MEV).cpu().numpy())
                all_targets.append((target * HA_TO_MEV).cpu().numpy())
    
    if num_samples == 0:
        return float('inf'), np.array([]), np.array([])
    
    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_targets = np.concatenate(all_targets) if all_targets else np.array([])
    
    return total_mae / num_samples, all_preds, all_targets


def plot_predictions(predictions, targets, epoch, mae, output_dir):
    """予測結果をプロット"""
    from sklearn.metrics import r2_score
    
    valid_mask = ~(np.isnan(predictions) | np.isnan(targets) | 
                   np.isinf(predictions) | np.isinf(targets))
    
    if valid_mask.sum() == 0:
        print(f"\n⚠️  Warning: No valid data for plotting at epoch {epoch}")
        return
    
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    ax1.scatter(targets, predictions, alpha=0.4, s=20, edgecolors='none')
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('True H (meV)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted H (meV)', fontsize=12, fontweight='bold')
    r2_stat = r2_score(targets, predictions)
    ax1.set_title(f'EGNN Baseline - Epoch {epoch}\nMAE: {mae:.2f} meV, R²: {r2_stat:.4f}', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    errors = predictions - targets
    ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Prediction Error (meV)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Error Distribution\nMean: {errors.mean():.2f} meV, Std: {errors.std():.2f} meV', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'H_baseline_predictions_epoch_{epoch}.png'), dpi=200)
    plt.close()


# =============================================
# メイン関数
# =============================================

def main():
    QM9_DIR = "/home/users/uchiyama/QM9_U0"
    
    # ★PFP版のグラフを使用（ベースラインは特徴抽出時にone-hotのみ使う）
    GRAPH_DIR = "/home/users/uchiyama/QM9_U0/graphs_U0_qm9_B3LYP_baseline_perfect"
    OUTPUT_DIR = "/home/users/uchiyama/QM9_U0/QM9_U0_training_baseline_from_base_perfect_0114"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("QM9 B3LYP H (Internal energy at 0K) 予測 - EGNN Baseline")
    print("EGNN公式実装（PFPなし）+ ターゲット標準化")
    print("PFP版グラフからone-hot特徴のみを抽出して使用")
    print("="*70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n🔹 データ読み込み中...")
    train_graphs = torch.load(os.path.join(GRAPH_DIR, "train_graphs.pt"), weights_only=False)
    val_graphs = torch.load(os.path.join(GRAPH_DIR, "val_graphs.pt"), weights_only=False)
    
    print(f"✓ Train: {len(train_graphs):,} molecules")
    print(f"✓ Val:   {len(val_graphs):,} molecules")
    
    sample = train_graphs[0]
    print(f"  ノード特徴次元: {sample.x.shape[1]}")
    print(f"  原子番号（index 256）: {sample.x[0, 256]}")  # 確認用
    print(f"  エッジ特徴: {sample.edge_attr.shape}")
    print(f"  座標: {sample.pos.shape}")
    print(f"  H: {sample.y.shape}")
    print(f"\n  → ベースラインでは原子番号のみを抽出してone-hot化 (5次元)")
    
    print("\n🔹 ターゲット標準化の統計量を計算中...")
    stats_loader = DataLoader(train_graphs, batch_size=96, shuffle=False, num_workers=0)
    mean, mad = compute_mean_mad(stats_loader, device)
    mean = torch.tensor(mean, dtype=torch.float32, device=device)
    mad = torch.tensor(mad, dtype=torch.float32, device=device)
    
    print("\n🔹 データローダー作成中...")
    batch_size = 32
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print("\n🔹 モデル初期化...")
    model = EGNN_Official(
        in_node_nf=5,  # ★one-hot (H, C, N, O, F)
        in_edge_nf=0,  # ★エッジ特徴なし
        hidden_nf=128,
        device=device,
        act_fn=nn.SiLU(),
        n_layers=7,
        coords_weight=1.0,
        attention=True,
        node_attr=True
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {num_params:,}")
    
    initial_lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-8)
    
    num_epochs = 1000
    scheduler = CosineAnnealingScheduler(
        optimizer, 
        max_epochs=num_epochs,
        eta_min=1e-7
    )
    
    print(f"\n{'='*70}")
    print("⚙️  EGNN公式実装準拠設定（Baseline: PFPなし）")
    print(f"{'='*70}")
    print(f"  Target:        H (Internal energy at 0K)")
    print(f"  Input:         5-dim one-hot (H,C,N,O,F) from PFP graph")
    print(f"  Edge feat:     None")
    print(f"  Hidden dim:    128")
    print(f"  Layers:        7")
    print(f"  Attention:     True")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Initial LR:    {initial_lr}")
    print(f"  Weight decay:  1e-8")
    print(f"  Pooling:       Sum (Extensive property)")
    print(f"{'='*70}")
    
    PAPER_MAE_MEV = 0.014 * 27211.4
    
    best_val_mae = float('inf')
    history = {
        'train_losses': [],
        'train_maes': [],
        'val_maes': [],
        'learning_rates': []
    }
    
    print(f"\n{'='*70}")
    print("🚀 Starting Training")
    print(f"{'='*70}\n")
    
    patience = 700
    patience_counter = 0
    min_improvement = 0.1
    
    for epoch in range(num_epochs):
        print(f"📊 Epoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        train_loss, train_mae = train_epoch_official(model, train_loader, optimizer, device, mean, mad)
        val_mae, val_preds, val_targets = validate_epoch_official(model, val_loader, device, mean, mad)
        
        current_lr = scheduler.step()
        
        history['train_losses'].append(train_loss)
        history['train_maes'].append(train_mae)
        history['val_maes'].append(val_mae)
        history['learning_rates'].append(current_lr)
        
        print(f"  Train: Loss={train_loss:.4f}, MAE={train_mae:.2f} meV")
        print(f"  Val:   MAE={val_mae:.2f} meV  |  LR={current_lr:.6f}")
        
        if val_mae < best_val_mae - min_improvement:
            improvement = best_val_mae - val_mae
            best_val_mae = val_mae
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'history': history,
                'mean': mean.item(),
                'mad': mad.item()
            }
            torch.save(checkpoint, os.path.join(OUTPUT_DIR, 'best_baseline_model.pth'))
            
            if epoch == 0:
                print(f"  ⭐ Initial best: {val_mae:.2f} meV")
            else:
                print(f"  ⭐ New best! Improved by {improvement:.2f} meV")
            
            if (epoch + 1) % 100 == 0:
                plot_predictions(val_preds, val_targets, epoch + 1, val_mae, OUTPUT_DIR)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered")
            break
        
        if (epoch + 1) % 100 == 0:
            torch.save(checkpoint, os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"  💾 Checkpoint saved")
        
        print()
    
    print("\n" + "="*70)
    print("📊 Training Complete")
    print("="*70)
    print(f"Best Val MAE: {best_val_mae:.2f} meV")
    print(f"Paper EGNN MAE (H): {PAPER_MAE_MEV:.2f} meV")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
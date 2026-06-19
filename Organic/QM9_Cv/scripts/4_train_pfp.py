#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QM9 B3LYP Cv (Heat capacity at 298.15K) 予測 - EGNN×PFP
EGNN公式実装の忠実な再現 + PFP記述子
単位: cal/(mol·K)
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
        
        # 座標更新は無効化
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        
        return h, coord, edge_attr


# =============================================
# EGNN公式実装
# =============================================

class EGNN_Official(nn.Module):
    """
    EGNN公式実装の忠実な再現
    QM9 Cv予測用
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
# データ前処理（公式実装形式に変換）
# =============================================

def prepare_batch_official(batch, device):
    """
    PyG形式のバッチを公式EGNN形式に変換
    """
    batch_size = batch.batch.max().item() + 1
    max_nodes = 0
    
    # 各グラフのノード数を計算
    node_counts = []
    for i in range(batch_size):
        mask = (batch.batch == i)
        node_counts.append(mask.sum().item())
        max_nodes = max(max_nodes, mask.sum().item())
    
    # パディング後のデータを準備
    total_nodes = batch_size * max_nodes
    
    # ノード特徴のパディング
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
    
    # エッジインデックスとエッジマスクの作成
    edges_list = []
    edge_attr_list = []
    edge_mask_list = []
    
    for i in range(batch_size):
        graph_mask = (batch.batch == i)
        graph_nodes = graph_mask.nonzero(as_tuple=True)[0]
        
        # このグラフのエッジを抽出
        edge_mask_graph = (graph_mask[batch.edge_index[0]] & 
                          graph_mask[batch.edge_index[1]])
        
        if edge_mask_graph.sum() > 0:
            graph_edges = batch.edge_index[:, edge_mask_graph]
            graph_edge_attr = batch.edge_attr[edge_mask_graph]
            
            # ノードインデックスを再マッピング
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
# Cv用訓練・評価関数（標準化なし）
# =============================================

def train_epoch_cv(model, loader, optimizer, device):
    """Cv用訓練（標準化なし）"""
    model.train()
    total_loss = 0
    total_mae = 0
    num_samples = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        
        h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, target = \
            prepare_batch_official(batch, device)
        
        optimizer.zero_grad()
        
        pred = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
        
        loss = nn.MSELoss()(pred, target)
        mae = torch.abs(pred - target).mean()
        
        # NaN/Infチェック
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  Warning: NaN/Inf loss detected, skipping batch")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if not torch.isnan(mae) and not torch.isinf(mae):
            total_loss += loss.item() * len(target)
            total_mae += mae.item() * len(target)
            num_samples += len(target)
    
    if num_samples == 0:
        return float('inf'), float('inf')
    
    return total_loss / num_samples, total_mae / num_samples


def validate_epoch_cv(model, loader, device):
    """Cv用検証（標準化なし）"""
    model.eval()
    total_mae = 0
    num_samples = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            batch = batch.to(device)
            
            h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, target = \
                prepare_batch_official(batch, device)
            
            pred = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
            
            # NaNチェック
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"\n⚠️  Warning: NaN/Inf detected in predictions!")
                continue
            
            mae = torch.abs(pred - target).mean()
            
            if not torch.isnan(mae) and not torch.isinf(mae):
                total_mae += mae.item() * len(target)
                num_samples += len(target)
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())
    
    if num_samples == 0:
        return float('inf'), np.array([]), np.array([])
    
    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_targets = np.concatenate(all_targets) if all_targets else np.array([])
    
    return total_mae / num_samples, all_preds, all_targets


def plot_predictions_cv(predictions, targets, epoch, mae, output_dir):
    """Cv予測結果をプロット"""
    from sklearn.metrics import r2_score
    
    # NaNチェック
    valid_mask = ~(np.isnan(predictions) | np.isnan(targets) | 
                   np.isinf(predictions) | np.isinf(targets))
    
    if valid_mask.sum() == 0:
        print(f"\n⚠️  Warning: No valid data for plotting at epoch {epoch}")
        return
    
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 散布図
    ax1 = axes[0]
    ax1.scatter(targets, predictions, alpha=0.4, s=20, edgecolors='none')
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('True Cv (cal/(mol·K))', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Cv (cal/(mol·K))', fontsize=12, fontweight='bold')
    r2_stat = r2_score(targets, predictions)
    ax1.set_title(f'EGNN×PFP (Official) - Epoch {epoch}\nMAE: {mae:.4f} cal/(mol·K), R²: {r2_stat:.4f}', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 誤差分布
    ax2 = axes[1]
    errors = predictions - targets
    ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Prediction Error (cal/(mol·K))', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Error Distribution\nMean: {errors.mean():.4f}, Std: {errors.std():.4f}', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cv_pfp_official_predictions_epoch_{epoch}.png'), dpi=200)
    plt.close()


# =============================================
# メイン関数
# =============================================

def main():
    # パス設定
    QM9_CV_DIR = "/home/users/uchiyama/QM9_Cv"
    GRAPH_DIR = os.path.join(QM9_CV_DIR, "graphs_Cv_qm9_B3LYP_pfp_v2")
    OUTPUT_DIR = os.path.join(QM9_CV_DIR, "training_pfp_Cv_egnn_1216")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("QM9 B3LYP Cv (Heat capacity at 298.15K) 予測 - EGNN×PFP")
    print("EGNN公式実装の忠実な再現 + PFP記述子")
    print("単位: cal/(mol·K)")
    print("="*70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # データ読み込み
    print("\n🔹 データ読み込み中...")
    train_graphs = torch.load(os.path.join(GRAPH_DIR, "train_graphs.pt"), weights_only=False)
    val_graphs = torch.load(os.path.join(GRAPH_DIR, "val_graphs.pt"), weights_only=False)
    
    print(f"✓ Train: {len(train_graphs):,} molecules")
    print(f"✓ Val:   {len(val_graphs):,} molecules")
    
    sample = train_graphs[0]
    print(f"\n📊 データ形状確認:")
    print(f"  ノード特徴: {sample.x.shape}")
    print(f"  エッジ特徴: {sample.edge_attr.shape}")
    print(f"  座標: {sample.pos.shape}")
    print(f"  Cv: {sample.y.shape}")
    
    # 訓練用ローダー作成
    print("\n🔹 データローダー作成中...")
    batch_size = 32
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # モデル初期化
    print("\n🔹 モデル初期化...")
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
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {num_params:,}")
    
    # オプティマイザとスケジューラ（準拠版）
    initial_lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-8)
    
    num_epochs = 1000
    scheduler = CosineAnnealingScheduler(
        optimizer, 
        max_epochs=num_epochs,
        eta_min=1e-7
    )
    
    print(f"\n{'='*70}")
    print("⚙️  EGNN公式実装準拠設定 + PFP記述子")
    print(f"{'='*70}")
    print(f"  Target:        Cv (Heat capacity at 298.15K)")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Initial LR:    {initial_lr}")
    print(f"  Weight decay:  1e-8")
    print(f"  Scheduler:     Cosine Annealing")
    print(f"  Loss:          MSE")
    print(f"  Hidden dim:    128")
    print(f"  Layers:        7")
    print(f"  Input:         261-dim (PFP256 + atom1 + geom4)")
    print(f"  Edge feat:     4-dim")
    print(f"  Coord update:  Disabled")
    print(f"  Residual:      True")
    print(f"  Attention:     False")
    print(f"  Node attr:     True")
    print(f"  Unit:          cal/(mol·K)")
    print(f"  Grad clip:     1.0")
    print(f"{'='*70}")
    
    # EGNN論文のCv MAE
    PAPER_MAE_CV = 0.026  # cal/(mol·K)
    
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
    
    # Early stopping
    patience = 100
    patience_counter = 0
    min_improvement = 0.001  # cal/(mol·K)
    
    for epoch in range(num_epochs):
        print(f"📊 Epoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        train_loss, train_mae = train_epoch_cv(model, train_loader, optimizer, device)
        val_mae, val_preds, val_targets = validate_epoch_cv(model, val_loader, device)
        
        current_lr = scheduler.step()
        
        history['train_losses'].append(train_loss)
        history['train_maes'].append(train_mae)
        history['val_maes'].append(val_mae)
        history['learning_rates'].append(current_lr)
        
        print(f"  Train: Loss={train_loss:.4f}, MAE={train_mae:.4f} cal/(mol·K)")
        print(f"  Val:   MAE={val_mae:.4f} cal/(mol·K)  |  LR={current_lr:.6f}")
        
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
                'model_type': 'egnn_pfp_official',
                'official_implementation': True,
                'unit': 'cal/(mol·K)'
            }
            torch.save(checkpoint, os.path.join(OUTPUT_DIR, 'best_cv_pfp_official_model.pth'))
            
            if epoch == 0:
                print(f"  ⭐ Initial best: {val_mae:.4f} cal/(mol·K)")
            else:
                print(f"  ⭐ New best! Improved by {improvement:.4f} cal/(mol·K)")
            
            if (epoch + 1) % 100 == 0:
                plot_predictions_cv(val_preds, val_targets, epoch + 1, val_mae, OUTPUT_DIR)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered")
            break
        
        if (epoch + 1) % 100 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'history': history,
                'model_type': 'egnn_pfp_official',
                'official_implementation': True,
                'unit': 'cal/(mol·K)'
            }
            torch.save(checkpoint, os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"  💾 Checkpoint saved")
        
        print()
    
    # 訓練完了
    print("\n" + "="*70)
    print("📊 Training Complete")
    print("="*70)
    print(f"Best Val MAE: {best_val_mae:.4f} cal/(mol·K)")
    print(f"Paper EGNN MAE (Cv): {PAPER_MAE_CV:.4f} cal/(mol·K)")
    
    improvement_ratio = (PAPER_MAE_CV - best_val_mae) / PAPER_MAE_CV * 100
    if improvement_ratio > 0:
        print(f"Improvement over paper: {improvement_ratio:.1f}%")
    else:
        print(f"Difference from paper: {abs(improvement_ratio):.1f}%")
    
    # 履歴プロット
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # MAE
    ax1 = axes[0, 0]
    ax1.plot(history['train_maes'], label='Train MAE', linewidth=2)
    ax1.plot(history['val_maes'], label='Val MAE', linewidth=2)
    ax1.axhline(y=PAPER_MAE_CV, color='red', linestyle='--', linewidth=2, 
                label=f'Paper EGNN ({PAPER_MAE_CV:.4f} cal/(mol·K))')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE (cal/(mol·K))', fontsize=12, fontweight='bold')
    ax1.set_title('EGNN×PFP (Official) - MAE (Cv)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2 = axes[0, 1]
    ax2.plot(history['train_losses'], label='Train Loss', linewidth=2, color='orange')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Learning Rate
    ax3 = axes[1, 0]
    ax3.plot(history['learning_rates'], linewidth=2, color='green')
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax3.set_title('LR Schedule (Cosine)', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 散布図
    ax4 = axes[1, 1]
    if len(val_preds) > 0 and len(val_targets) > 0:
        from sklearn.metrics import r2_score
        ax4.scatter(val_targets, val_preds, alpha=0.4, s=15, edgecolors='none')
        min_val = min(val_targets.min(), val_preds.min())
        max_val = max(val_targets.max(), val_preds.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        r2_val = r2_score(val_targets, val_preds)
        ax4.set_xlabel('True Cv (cal/(mol·K))', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Predicted Cv (cal/(mol·K))', fontsize=12, fontweight='bold')
        ax4.set_title(f'EGNN×PFP Official Results (MAE: {best_val_mae:.4f} cal/(mol·K), R²: {r2_val:.4f})', 
                    fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No validation data available', 
                 ha='center', va='center', fontsize=14)
        ax4.set_xlabel('True Cv (cal/(mol·K))', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Predicted Cv (cal/(mol·K))', fontsize=12, fontweight='bold')
        ax4.set_title('EGNN×PFP Official Results', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_cv_pfp_official.png'), dpi=200)
    plt.close()
    
    # 統計情報を保存
    stats = {
        'dataset': 'QM9',
        'functional': 'B3LYP/6-31G(2df,p)',
        'target': 'Cv (Heat capacity at 298.15K)',
        'model_version': 'egnn_pfp_official_implementation',
        'official_implementation': True,
        'reference': 'EGNN (Satorras et al., 2021) - Official implementation',
        'training': {
            'best_val_mae': float(best_val_mae),
            'paper_egnn_mae': PAPER_MAE_CV,
            'unit': 'cal/(mol·K)',
            'epochs_trained': epoch + 1,
            'num_parameters': num_params,
            'batch_size': batch_size,
            'initial_lr': initial_lr,
            'weight_decay': 1e-8,
            'grad_clip': 1.0
        },
        'model_config': {
            'input_features': '261-dim (PFP256 + atomic_num1 + geom4)',
            'edge_features': '4-dim',
            'hidden_dim': 128,
            'num_layers': 7,
            'coord_update': False,
            'coords_weight': 1.0,
            'residual': True,
            'attention': False,
            'node_attr': True,
            'pfp_enabled': True,
            'target_normalized': False
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(OUTPUT_DIR, 'training_stats_cv_pfp_official.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*70}")
    print("🎉 EGNN×PFP Official Training Complete (Cv)!")
    print(f"{'='*70}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
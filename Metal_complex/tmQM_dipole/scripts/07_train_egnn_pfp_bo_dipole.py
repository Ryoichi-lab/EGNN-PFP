#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tmQM 金属錯体 双極子モーメント予測 - EGNN×PFP+BO
EGNN原著論文準拠 + Matlantis PFP記述子 + 結合次数(BO)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_scatter import scatter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
import math
from sklearn.metrics import r2_score

# =============================================
# EGNN Layer（エッジ特徴対応版）
# =============================================

class EGNNLayerWithEdgeFeatures(MessagePassing):
    """EGNN原著論文準拠 + エッジ特徴（Bond Order含む）"""
    def __init__(self, hidden_dim, edge_dim, activation='swish'):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

        self.edge_inference = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, edge_index, edge_attr, pos):
        h_updated, coord_update = self.propagate(
            edge_index, h=h, edge_attr=edge_attr, pos=pos
        )
        h_new = h + h_updated
        pos_new = pos + coord_update
        return h_new, pos_new

    def message(self, h_i, h_j, edge_attr, pos_i, pos_j):
        rel_pos = pos_i - pos_j
        dist_sq = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)
        message_input = torch.cat([h_i, h_j, dist_sq, edge_attr], dim=-1)
        message = self.message_mlp(message_input)
        edge_weight = self.edge_inference(message)
        return message * edge_weight

    def propagate(self, edge_index, h, edge_attr, pos):
        out = super().propagate(edge_index, h=h, edge_attr=edge_attr, pos=pos)
        row, col = edge_index
        h_i, h_j = h[row], h[col]
        pos_i, pos_j = pos[row], pos[col]
        rel_pos_ij = pos_i - pos_j
        dist_sq = torch.sum(rel_pos_ij ** 2, dim=-1, keepdim=True)
        message_input = torch.cat([h_i, h_j, dist_sq, edge_attr], dim=-1)
        message = self.message_mlp(message_input)
        coord_weights = self.coord_mlp(message)
        coord_update_edges = rel_pos_ij * coord_weights

        coord_update = scatter(coord_update_edges, row, dim=0,
                               dim_size=pos.size(0), reduce='add')
        count = scatter(torch.ones(edge_index.size(1), 1, device=pos.device),
                        row, dim=0, dim_size=pos.size(0), reduce='add')
        coord_update = coord_update / (count + 1e-8)
        return out, coord_update

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        out = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='add')
        count = scatter(torch.ones(inputs.size(0), 1, device=inputs.device),
                        index, dim=0, dim_size=dim_size, reduce='add')
        return out / (count + 1e-8)

    def update(self, aggr_out, h):
        update_input = torch.cat([h, aggr_out], dim=-1)
        return self.node_mlp(update_input)

# =============================================
# EGNN×PFP+BO モデル
# =============================================

class EGNN_PFP_BO(nn.Module):
    """
    EGNN×PFP+BO
    入力: PFP記述子(256) + 原子番号(1) + 幾何特徴(4) = 261次元
    エッジ: 距離 + PFP相互作用 + Bond Order = 5次元
    """
    def __init__(self, input_dim=261, hidden_dim=128, edge_dim=5, num_layers=7):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        self.egnn_layers = nn.ModuleList([
            EGNNLayerWithEdgeFeatures(hidden_dim=hidden_dim,
                                      edge_dim=hidden_dim // 4)
            for _ in range(num_layers)
        ])
        self.dipole_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch):
        h = self.node_encoder(batch.x)
        edge_feat = self.edge_encoder(batch.edge_attr)
        pos = batch.pos
        for layer in self.egnn_layers:
            h, pos = layer(h, batch.edge_index, edge_feat, pos)
        node_dipoles = self.dipole_head(h)
        molecular_dipole = global_add_pool(node_dipoles, batch.batch)
        return {
            'molecular_dipole': molecular_dipole,
            'node_embeddings': h,
            'updated_positions': pos
        }

# =============================================
# Cosine Annealing Scheduler
# =============================================

class CosineAnnealingScheduler:
    def __init__(self, optimizer, max_epochs, eta_min=1e-7):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.epoch = 0

    def step(self):
        progress = self.epoch / self.max_epochs
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.epoch += 1
        return lr

# =============================================
# 訓練関数・検証関数
# =============================================

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, total_mae = 0, 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        pred = outputs['molecular_dipole'].view(-1)
        true = batch.y.view(-1)
        loss = nn.MSELoss()(pred, true)
        mae = torch.mean(torch.abs(pred - true))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_mae += mae.item()
    n = len(dataloader)
    return total_loss/n, total_mae/n

def validate_epoch(model, dataloader, device):
    model.eval()
    total_mae = 0
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            batch = batch.to(device)
            out = model(batch)
            p = out['molecular_dipole'].view(-1)
            t = batch.y.view(-1)
            total_mae += torch.mean(torch.abs(p - t)).item()
            preds.append(p.cpu().numpy())
            trues.append(t.cpu().numpy())
    return total_mae / len(dataloader), np.concatenate(preds), np.concatenate(trues)

def plot_predictions(predictions, targets, epoch, mae, output_dir, split_name="Val"):
    """予測結果プロット"""
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.4, s=15, edgecolors='none')
    
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect')
    
    r2 = r2_score(targets, predictions)
    
    textstr = f'EGNN×PFP+BO (tmQM)\nEpoch: {epoch}\n{split_name} MAE: {mae:.4f} D\nR²: {r2:.4f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    plt.xlabel('True Dipole (D)', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Dipole (D)', fontsize=14, fontweight='bold')
    plt.title(f'EGNN×PFP+BO - {split_name} - Epoch {epoch}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'bo_predictions_{split_name.lower()}_epoch_{epoch}.png'), dpi=200)
    plt.close()

# =============================================
# メイン処理
# =============================================

def main():
    TMQM_DIR = "/home/users/uchiyama/tmQM_dipole"
    GRAPH_DIR = os.path.join(TMQM_DIR, "graphs_tmQM_pfp_dipole_bo2")
    OUTPUT_DIR = os.path.join(TMQM_DIR, "training_pfp_bo_egnn_v3_0124")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(16)

    print("="*70)
    print("tmQM 金属錯体 双極子予測 - EGNN×PFP+BO")
    print("EGNN原著論文準拠 + Matlantis PFP + Bond Order")
    print("="*70)
    print(f"Device: {device}")
    print(f"CPU threads: {torch.get_num_threads()}")

    train_graphs = torch.load(os.path.join(GRAPH_DIR, "train_graphs.pt"))
    val_graphs = torch.load(os.path.join(GRAPH_DIR, "val_graphs.pt"))
    test_graphs = torch.load(os.path.join(GRAPH_DIR, "test_graphs.pt"))
    print(f"\n✓ Train: {len(train_graphs):,} | Val: {len(val_graphs):,} | Test: {len(test_graphs):,}")

    sample = train_graphs[0]
    print(f"\n📊 データ形状確認:")
    print(f"  ノード特徴: {sample.x.shape}")
    print(f"  エッジ特徴: {sample.edge_attr.shape}")
    print(f"  座標: {sample.pos.shape}")

    batch_size = 16
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=8)

    model = EGNN_PFP_BO(input_dim=261, hidden_dim=128, edge_dim=5, num_layers=7).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Parameters: {num_params:,}")
    
    initial_lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-16)
    num_epochs = 1000
    scheduler = CosineAnnealingScheduler(optimizer, max_epochs=num_epochs, eta_min=1e-7)

    print(f"\n{'='*70}")
    print("⚙️  訓練設定")
    print(f"{'='*70}")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Initial LR:    {initial_lr}")
    print(f"  Scheduler:     Cosine Annealing")
    print(f"  Hidden dim:    128")
    print(f"  Layers:        7")
    print(f"  Input:         261-dim (PFP256 + atom1 + geom4)")
    print(f"  Edge feat:     5-dim (dist + PFP + BO)")
    print(f"{'='*70}")

    best_val_mae = float('inf')
    history = {
        'train_losses': [],
        'train_maes': [],
        'val_maes': [],
        'learning_rates': []
    }

    print(f"\n{'='*70}")
    print("🚀 Starting Training - EGNN×PFP+BO")
    print(f"{'='*70}\n")

    for epoch in range(num_epochs):
        print(f"📊 Epoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, device)
        val_mae, val_preds, val_targets = validate_epoch(model, val_loader, device)
        current_lr = scheduler.step()
        
        history['train_losses'].append(train_loss)
        history['train_maes'].append(train_mae)
        history['val_maes'].append(val_mae)
        history['learning_rates'].append(current_lr)
        
        print(f"  Train: Loss={train_loss:.4f}, MAE={train_mae:.4f} D")
        print(f"  Val:   MAE={val_mae:.4f} D  |  LR={current_lr:.6f}")

        if val_mae < best_val_mae:
            improvement = best_val_mae - val_mae
            best_val_mae = val_mae
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'history': history,
                'model_type': 'egnn_pfp_bo_tmqm'
            }
            torch.save(checkpoint, os.path.join(OUTPUT_DIR, 'best_model_bo.pth'))
            
            if epoch == 0:
                print(f"  ⭐ Initial best: {val_mae:.4f} D")
            else:
                print(f"  ⭐ New best! Improved by {improvement:.4f} D")
            
            if (epoch + 1) % 50 == 0:
                plot_predictions(val_preds, val_targets, epoch + 1, val_mae, OUTPUT_DIR, "Val")
        
        if (epoch + 1) % 50 == 0:
            torch.save(checkpoint, os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"  💾 Checkpoint saved")
        
        print()

    # 最終テスト評価
    print("\n" + "="*70)
    print("🧪 Final Test Evaluation")
    print("="*70)
    
    best_checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model_bo.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_mae, test_preds, test_targets = validate_epoch(model, test_loader, device)
    
    print(f"\nTest MAE: {test_mae:.4f} D")
    plot_predictions(test_preds, test_targets, best_checkpoint['epoch'], test_mae, OUTPUT_DIR, "Test")

    # 履歴プロット
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    ax1.plot(history['train_maes'], label='Train MAE', linewidth=2)
    ax1.plot(history['val_maes'], label='Val MAE', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE (D)', fontsize=12, fontweight='bold')
    ax1.set_title('EGNN×PFP+BO - MAE', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(history['train_losses'], label='Train Loss', linewidth=2, color='orange')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(history['learning_rates'], linewidth=2, color='green')
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax3.set_title('LR Schedule (Cosine)', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.scatter(test_targets, test_preds, alpha=0.4, s=15, edgecolors='none')
    min_val = min(test_targets.min(), test_preds.min())
    max_val = max(test_targets.max(), test_preds.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    r2 = r2_score(test_targets, test_preds)
    ax4.set_xlabel('True (D)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Predicted (D)', fontsize=12, fontweight='bold')
    ax4.set_title(f'Test Results (MAE: {test_mae:.4f} D, R²: {r2:.4f})', 
                fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_bo.png'), dpi=200)
    plt.close()

    # 統計保存
    stats = {
        'dataset': 'tmQM',
        'model_version': 'egnn_pfp_bo',
        'training': {
            'best_val_mae': float(best_val_mae),
            'test_mae': float(test_mae),
            'epochs_trained': epoch + 1,
            'num_parameters': num_params
        },
        'model_config': {
            'input_dim': 261,
            'edge_dim': 5,
            'hidden_dim': 128,
            'num_layers': 7
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(OUTPUT_DIR, 'training_stats_bo.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*70}")
    print("🎉 Training Complete!")
    print(f"{'='*70}")
    print(f"Best Val MAE:  {best_val_mae:.4f} D")
    print(f"Test MAE:      {test_mae:.4f} D")
    print(f"Output:        {OUTPUT_DIR}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
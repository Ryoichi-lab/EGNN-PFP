#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QM9 B3LYP HOMO予測 - ベースラインEGNN
EGNN原著論文完全準拠（PFP記述子なし）
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

# =============================================
# EGNN Layer（論文準拠）
# =============================================

class EGNNLayer(MessagePassing):
    """EGNN原著論文 Section 5.3 QM9実験準拠"""
    def __init__(self, hidden_dim, activation='swish'):
        super().__init__(aggr='add')
        
        self.hidden_dim = hidden_dim
        
        # φe: メッセージ関数（論文 式3）
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # φx: 座標更新関数（論文 式4）
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
        # φinf: エッジ推論（論文 Section 3.3）
        self.edge_inference = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # φh: ノード更新関数（論文 式6）
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, h, edge_index, pos):
        # メッセージパッシング + 座標更新
        h_updated, coord_update = self.propagate(
            edge_index, 
            h=h, 
            pos=pos
        )
        
        # ノード更新
        h_new = h + h_updated
        
        # 座標更新
        pos_new = pos + coord_update
        
        return h_new, pos_new
    
    def message(self, h_i, h_j, pos_i, pos_j):
        """論文 式(3)"""
        rel_pos = pos_i - pos_j
        dist_sq = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)
        
        message_input = torch.cat([h_i, h_j, dist_sq], dim=-1)
        message = self.message_mlp(message_input)
        
        # エッジ推論
        edge_weight = self.edge_inference(message)
        message = message * edge_weight
        
        return message
    
    def propagate(self, edge_index, h, pos):
        """座標更新を含むメッセージパッシング"""
        # ノード更新
        out = super().propagate(edge_index, h=h, pos=pos)
        
        # 座標更新（論文 式4）
        row, col = edge_index
        h_i, h_j = h[row], h[col]
        pos_i, pos_j = pos[row], pos[col]
        
        rel_pos_ij = pos_i - pos_j
        dist_sq = torch.sum(rel_pos_ij ** 2, dim=-1, keepdim=True)
        
        message_input = torch.cat([h_i, h_j, dist_sq], dim=-1)
        message = self.message_mlp(message_input)
        
        # 座標更新の重み
        coord_weights = self.coord_mlp(message)
        coord_update_edges = rel_pos_ij * coord_weights
        
        # 集約
        coord_update = scatter(
            coord_update_edges, 
            row, 
            dim=0, 
            dim_size=pos.size(0), 
            reduce='add'
        )
        
        # 正規化（論文 式5のC）
        count = scatter(
            torch.ones(edge_index.size(1), 1, device=pos.device),
            row,
            dim=0,
            dim_size=pos.size(0),
            reduce='add'
        )
        coord_update = coord_update / (count + 1e-8)
        
        return out, coord_update
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """論文 式(5)"""
        out = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='add')
        
        count = scatter(
            torch.ones(inputs.size(0), 1, device=inputs.device),
            index,
            dim=0,
            dim_size=dim_size,
            reduce='add'
        )
        
        out = out / (count + 1e-8)
        return out
    
    def update(self, aggr_out, h):
        """論文 式(6)"""
        update_input = torch.cat([h, aggr_out], dim=-1)
        h_new = self.node_mlp(update_input)
        return h_new


# =============================================
# ベースラインEGNN（PFPなし）
# =============================================

class BaselineEGNN(nn.Module):
    """
    ベースラインEGNN - EGNN原著論文準拠
    入力: 原子タイプのみ（5次元 one-hot: H, C, N, O, F）
    """
    def __init__(self, 
                 input_dim=5,
                 hidden_dim=128,
                 num_layers=7):
        super().__init__()
        
        # ノードエンコーダ
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # EGNN層（7層 - 論文準拠）
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim=hidden_dim, activation='swish')
            for _ in range(num_layers)
        ])
        
        # HOMO予測ヘッド（論文準拠）
        self.homo_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, batch):
        # 原子タイプone-hotエンコーディング
        x = self._create_atom_features(batch)
        
        # エンコーディング
        h = self.node_encoder(x)
        pos = batch.pos
        
        # EGNN層
        for layer in self.egnn_layers:
            h, pos = layer(h, batch.edge_index, pos)
        
        # ノードごとのHOMO寄与
        node_homo = self.homo_head(h)
        
        # 分子全体のHOMO（sum pooling - 論文準拠）
        molecular_homo = global_add_pool(node_homo, batch.batch)
        
        return {
            'molecular_homo': molecular_homo,
            'node_embeddings': h,
            'updated_positions': pos
        }
    
    def _create_atom_features(self, batch):
        """
        原子タイプのone-hotエンコーディング（論文準拠）
        H=1, C=6, N=7, O=8, F=9
        """
        # batch.xの最初の5次元が原子タイプone-hot（ベースラインデータの場合）
        if batch.x.size(1) == 5:
            return batch.x
        else:
            # もしPFP付きデータの場合、最初の256次元から原子番号を抽出
            atomic_nums = torch.argmax(batch.x[:, :256], dim=1)
            features = torch.zeros(atomic_nums.size(0), 5, device=batch.x.device)
            
            # H=1→0, C=6→1, N=7→2, O=8→3, F=9→4
            atom_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
            for i, z in enumerate(atomic_nums):
                z_val = z.item()
                if z_val in atom_map:
                    features[i, atom_map[z_val]] = 1
            
            return features


# =============================================
# Cosine Annealing Scheduler（論文準拠）
# =============================================

class CosineAnnealingScheduler:
    """EGNN論文のCosine Decay"""
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
# 訓練・評価関数
# =============================================

def train_epoch(model, loader, optimizer, device):
    """1エポックの訓練"""
    model.train()
    total_loss = 0
    total_mae = 0
    num_samples = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        outputs = model(batch)
        pred = outputs['molecular_homo'].squeeze()
        target = batch.y.squeeze()
        
        loss = nn.MSELoss()(pred, target)
        mae = torch.abs(pred - target).mean()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch.y)
        total_mae += mae.item() * len(batch.y)
        num_samples += len(batch.y)
    
    return total_loss / num_samples, total_mae / num_samples


def validate_epoch(model, loader, device):
    """1エポックの評価"""
    model.eval()
    total_mae = 0
    num_samples = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            batch = batch.to(device)
            
            outputs = model(batch)
            pred = outputs['molecular_homo'].squeeze()
            target = batch.y.squeeze()
            
            mae = torch.abs(pred - target).mean()
            
            total_mae += mae.item() * len(batch.y)
            num_samples += len(batch.y)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    return total_mae / num_samples, all_preds, all_targets


def plot_predictions(predictions, targets, epoch, mae, output_dir):
    """予測結果のプロット"""
    from sklearn.metrics import r2_score
    
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.4, s=20, edgecolors='none')
    
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    r2 = r2_score(targets, predictions)
    
    textstr = f'Baseline EGNN\nEpoch: {epoch}\nMAE: {mae:.4f} eV\nR²: {r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    plt.xlabel('True HOMO (eV)', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted HOMO (eV)', fontsize=14, fontweight='bold')
    plt.title(f'Baseline EGNN (No PFP) - Epoch {epoch}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'baseline_predictions_epoch_{epoch}.png'), dpi=200)
    plt.close()


# =============================================
# メイン関数
# =============================================

def main():
    # 絶対パス設定
    QM9_HOMO_DIR = "/home/users/uchiyama/QM9_HOMO_LUMO"
    GRAPH_DIR = os.path.join(QM9_HOMO_DIR, "graphs_homo_qm9_B3LYP_baseline")
    OUTPUT_DIR = "/home/users/uchiyama/QM9_HOMO/training_baseline_HOMO_egnn"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("QM9 B3LYP HOMO予測 - ベースラインEGNN（PFPなし）")
    print("EGNN原著論文完全準拠（Cormorant split）")
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
    
    # DataLoader（論文準拠 batch_size=96）
    batch_size = 96
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # モデル初期化（論文準拠）
    print("\n🔹 モデル初期化...")
    model = BaselineEGNN(
        input_dim=5,
        hidden_dim=128,
        num_layers=7
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {num_params:,}")
    
    # 最適化設定（論文準拠）
    initial_lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-16)
    
    num_epochs = 1000
    scheduler = CosineAnnealingScheduler(
        optimizer, 
        max_epochs=num_epochs,
        eta_min=1e-7
    )
    
    print(f"\n{'='*70}")
    print("⚙️  EGNN原著論文準拠設定")
    print(f"{'='*70}")
    print(f"  Epochs:        1000")
    print(f"  Batch size:    96")
    print(f"  Initial LR:    5e-4")
    print(f"  Weight decay:  1e-16")
    print(f"  Scheduler:     Cosine Annealing")
    print(f"  Loss:          MSE")
    print(f"  Hidden dim:    128")
    print(f"  Layers:        7")
    print(f"  Input:         5-dim one-hot (H,C,N,O,F)")
    print(f"  Coord update:  Enabled")
    print(f"  Target:        HOMO Energy")
    print(f"{'='*70}")
    
    # 訓練ループ
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
        
        print(f"  Train: Loss={train_loss:.4f}, MAE={train_mae:.4f} eV")
        print(f"  Val:   MAE={val_mae:.4f} eV  |  LR={current_lr:.6f}")
        
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
                'model_type': 'baseline_egnn',
                'paper_compliant': True
            }
            torch.save(checkpoint, os.path.join(OUTPUT_DIR, 'best_baseline_model.pth'))
            
            if epoch == 0:
                print(f"  ⭐ Initial best: {val_mae:.4f} eV")
            else:
                print(f"  ⭐ New best! Improved by {improvement:.4f} eV")
            
            if (epoch + 1) % 100 == 0:
                plot_predictions(val_preds, val_targets, epoch + 1, val_mae, OUTPUT_DIR)
        
        if (epoch + 1) % 100 == 0:
            torch.save(checkpoint, os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"  💾 Checkpoint saved")
        
        print()
    
    # 結果保存
    print("\n" + "="*70)
    print("📊 Training Complete")
    print("="*70)
    print(f"Best Val MAE: {best_val_mae:.4f} eV")
    print(f"Paper EGNN MAE (HOMO): 0.041 eV")
    
    # 履歴プロット
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    ax1.plot(history['train_maes'], label='Train MAE', linewidth=2)
    ax1.plot(history['val_maes'], label='Val MAE', linewidth=2)
    ax1.axhline(y=0.041, color='red', linestyle='--', linewidth=2, label='Paper EGNN (0.041 eV)')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE (eV)', fontsize=12, fontweight='bold')
    ax1.set_title('Baseline EGNN - MAE', fontsize=14, fontweight='bold')
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
    
    from sklearn.metrics import r2_score
    ax4 = axes[1, 1]
    ax4.scatter(val_targets, val_preds, alpha=0.4, s=15, edgecolors='none')
    min_val = min(val_targets.min(), val_preds.min())
    max_val = max(val_targets.max(), val_preds.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    r2 = r2_score(val_targets, val_preds)
    ax4.set_xlabel('True HOMO (eV)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Predicted HOMO (eV)', fontsize=12, fontweight='bold')
    ax4.set_title(f'Baseline Results (MAE: {best_val_mae:.4f} eV, R²: {r2:.4f})', 
                fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_baseline.png'), dpi=200)
    plt.close()
    
    # 統計保存
    stats = {
        'dataset': 'QM9',
        'functional': 'B3LYP/6-31G(2df,p)',
        'target': 'HOMO_energy',
        'model_version': 'baseline_egnn_no_pfp',
        'paper_compliant': True,
        'reference': 'EGNN (Satorras et al., 2021) - Cormorant split',
        'training': {
            'best_val_mae': float(best_val_mae),
            'paper_egnn_mae': 0.041,
            'unit': 'eV',
            'epochs_trained': epoch + 1,
            'num_parameters': num_params,
            'batch_size': batch_size,
            'initial_lr': initial_lr
        },
        'model_config': {
            'input_features': '5-dim one-hot (H,C,N,O,F)',
            'hidden_dim': 128,
            'num_layers': 7,
            'coord_update': True,
            'edge_inference': True
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(OUTPUT_DIR, 'training_stats_baseline.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*70}")
    print("🎉 Baseline Training Complete!")
    print(f"{'='*70}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
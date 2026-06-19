#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tmQM 金属錯体 LUMO予測 - EGNN×PFP
EGNN原著論文準拠 + Matlantis PFP記述子
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
# EGNN Layer（論文準拠 + エッジ特徴）
# =============================================

class EGNNLayerWithEdgeFeatures(MessagePassing):
    """EGNN原著論文準拠 + エッジ特徴対応"""
    def __init__(self, hidden_dim, edge_dim, activation='swish'):
        super().__init__(aggr='add')
        
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        
        # φe: メッセージ関数（エッジ特徴を含む）
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # φx: 座標更新関数
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
        # φinf: エッジ推論
        self.edge_inference = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # φh: ノード更新関数
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, h, edge_index, edge_attr, pos):
        h_updated, coord_update = self.propagate(
            edge_index, 
            h=h,
            edge_attr=edge_attr,
            pos=pos
        )
        
        h_new = h + h_updated
        pos_new = pos + coord_update
        
        return h_new, pos_new
    
    def message(self, h_i, h_j, edge_attr, pos_i, pos_j):
        """論文 式(3) + エッジ特徴"""
        rel_pos = pos_i - pos_j
        dist_sq = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)
        
        message_input = torch.cat([h_i, h_j, dist_sq, edge_attr], dim=-1)
        message = self.message_mlp(message_input)
        
        edge_weight = self.edge_inference(message)
        message = message * edge_weight
        
        return message
    
    def propagate(self, edge_index, h, edge_attr, pos):
        """座標更新を含むメッセージパッシング"""
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
        
        coord_update = scatter(
            coord_update_edges, 
            row, 
            dim=0, 
            dim_size=pos.size(0), 
            reduce='add'
        )
        
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
        
        count = scatter(torch.ones(inputs.size(0), 1, device=inputs.device),
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
# EGNN×PFP モデル（LUMO-LUMOギャップ用）
# =============================================
class EGNN_PFP_LUMO(nn.Module):
    """
    EGNN×PFP - LUMO予測用
    入力: PFP記述子(256) + 原子番号(1) + 幾何特徴(4) = 261次元
    """
    def __init__(self, 
                 input_dim=261,
                 hidden_dim=128,
                 edge_dim=4,
                 num_layers=7):
        super().__init__()
        
        # ノードエンコーダ
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # エッジエンコーダ
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # EGNN層（7層 - 論文準拠）
        self.egnn_layers = nn.ModuleList([
            EGNNLayerWithEdgeFeatures(
                hidden_dim=hidden_dim,
                edge_dim=hidden_dim // 4,
                activation='swish'
            )
            for _ in range(num_layers)
        ])
        
        # LUMO予測ヘッド
        self.LUMO_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, batch):
        # エンコーディング
        h = self.node_encoder(batch.x)
        edge_feat = self.edge_encoder(batch.edge_attr)
        pos = batch.pos
        
        # EGNN層
        for layer in self.egnn_layers:
            h, pos = layer(h, batch.edge_index, edge_feat, pos)
        
        # グラフレベルの集約（mean pooling - エネルギー関連なので平均を使用）
        graph_embedding = scatter(h, batch.batch, dim=0, reduce='mean')
        
        # LUMO予測
        LUMO_pred = self.LUMO_head(graph_embedding)
        
        return {
            'LUMO_energy': LUMO_pred,
            'node_embeddings': h,
            'updated_positions': pos
        }


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
# 訓練・検証関数
# =============================================

def train_epoch(model, dataloader, optimizer, device):
    """1エポックの訓練"""
    model.train()
    total_loss = 0
    total_mae = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        LUMO_pred = outputs['LUMO_energy'].view(-1)
        LUMO_true = batch.y.view(-1)
        
        # MSE Loss（論文準拠）
        loss = nn.MSELoss()(LUMO_pred, LUMO_true)
        mae = torch.mean(torch.abs(LUMO_pred - LUMO_true))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mae += mae.item()
        num_batches += 1
    
    return total_loss / num_batches, total_mae / num_batches


def validate_epoch(model, dataloader, device):
    """検証"""
    model.eval()
    total_mae = 0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            batch = batch.to(device)
            
            outputs = model(batch)
            LUMO_pred = outputs['LUMO_energy'].view(-1)
            LUMO_true = batch.y.view(-1)
            
            mae = torch.mean(torch.abs(LUMO_pred - LUMO_true))
            
            total_mae += mae.item()
            num_batches += 1
            
            all_predictions.append(LUMO_pred.cpu().numpy())
            all_targets.append(LUMO_true.cpu().numpy())
    
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    return total_mae / num_batches, predictions, targets


def plot_predictions(predictions, targets, epoch, mae, output_dir, split_name="Val"):
    """予測結果プロット"""
    from sklearn.metrics import r2_score
    
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.4, s=15, edgecolors='none')
    
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect')
    
    r2 = r2_score(targets, predictions)
    
    textstr = f'EGNN×PFP (tmQM)\nEpoch: {epoch}\n{split_name} MAE: {mae:.4f} eV\nR²: {r2:.4f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

# 変更後
    plt.xlabel('True LUMO (eV)', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted LUMO (eV)', fontsize=14, fontweight='bold')
    plt.title(f'EGNN×PFP - {split_name} - Epoch {epoch}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'pfp_predictions_{split_name.lower()}_epoch_{epoch}.png'), dpi=200)
    plt.close()


# =============================================
# メイン関数
# =============================================

def main():
    # 絶対パス設定
    TMQM_DIR = "/home/users/uchiyama/tmQM_dipole"
    GRAPH_DIR = os.path.join(TMQM_DIR, "graphs_tmQM_pfp_lumo")
    OUTPUT_DIR = os.path.join(TMQM_DIR, "training_pfp_egnn_lumo")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # CPU最適化
    torch.set_num_threads(16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("tmQM 金属錯体 LUMO予測 - EGNN×PFP")
    print("EGNN原著論文準拠 + Matlantis PFP記述子")
    print("="*70)
    print(f"Device: {device}")
    print(f"CPU threads: {torch.get_num_threads()}")
    
    # データ読み込み
    print("\n🔹 データ読み込み中...")
    train_graphs = torch.load(os.path.join(GRAPH_DIR, "train_graphs.pt"), weights_only=False)
    val_graphs = torch.load(os.path.join(GRAPH_DIR, "val_graphs.pt"), weights_only=False)
    test_graphs = torch.load(os.path.join(GRAPH_DIR, "test_graphs.pt"), weights_only=False)
    
    print(f"✓ Train: {len(train_graphs):,} molecules")
    print(f"✓ Val:   {len(val_graphs):,} molecules")
    print(f"✓ Test:  {len(test_graphs):,} molecules")
    
    # データ確認
    sample = train_graphs[0]
    print(f"\n📊 データ形状確認:")
    print(f"  ノード特徴: {sample.x.shape}")
    print(f"  エッジ特徴: {sample.edge_attr.shape}")
    print(f"  座標: {sample.pos.shape}")
    print(f"  LUMO: {sample.y.shape}")
    
    # DataLoader
    batch_size = 16  # CPUなので少し大きめ
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=8)
    
    # モデル初期化（論文準拠）
    print("\n🔹 モデル初期化...")
    model = EGNN_PFP_LUMO(
        input_dim=261,
        hidden_dim=128,
        edge_dim=4,
        num_layers=7
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {num_params:,}")
    
    # 最適化設定（論文準拠）
    initial_lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-16)
    
    num_epochs = 500  # tmQMは大規模なので500エポック
    scheduler = CosineAnnealingScheduler(
        optimizer, 
        max_epochs=num_epochs,
        eta_min=1e-7
    )
    
    print(f"\n{'='*70}")
    print("⚙️  EGNN原著論文準拠設定 + Matlantis PFP")
    print(f"{'='*70}")
    print(f"  Dataset:       tmQM (金属錯体)")
    print(f"  Target:        LUMO Energy")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Initial LR:    {initial_lr}")
    print(f"  Weight decay:  1e-16")
    print(f"  Scheduler:     Cosine Annealing")
    print(f"  Loss:          MSE")
    print(f"  Hidden dim:    128")
    print(f"  Layers:        7")
    print(f"  Input:         261-dim (PFP256 + atom1 + geom4)")
    print(f"  Edge feat:     4-dim")
    print(f"  Coord update:  Enabled")
    print(f"  Pooling:       Mean (for energy-related property)")
    print(f"  Split:         Stratified by metal")
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
    print("🚀 Starting Training - tmQM Metal Complexes (LUMO)")
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
                'model_type': 'egnn_pfp_tmqm_LUMO',
                'dataset': 'tmQM_metal_complexes'
            }
            torch.save(checkpoint, os.path.join(OUTPUT_DIR, 'best_pfp_model.pth'))
            
            if epoch == 0:
                print(f"  ⭐ Initial best: {val_mae:.4f} eV")
            else:
                print(f"  ⭐ New best! Improved by {improvement:.4f} eV")
            
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
    
    # ベストモデルをロード
    best_checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_pfp_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_mae, test_preds, test_targets = validate_epoch(model, test_loader, device)
    
    print(f"\nTest MAE: {test_mae:.4f} eV")
    plot_predictions(test_preds, test_targets, best_checkpoint['epoch'], test_mae, OUTPUT_DIR, "Test")
    
    # 履歴プロット
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    ax1.plot(history['train_maes'], label='Train MAE', linewidth=2)
    ax1.plot(history['val_maes'], label='Val MAE', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE (eV)', fontsize=12, fontweight='bold')
    ax1.set_title('EGNN×PFP (tmQM LUMO) - MAE', fontsize=14, fontweight='bold')
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
    ax4.scatter(test_targets, test_preds, alpha=0.4, s=15, edgecolors='none')
    min_val = min(test_targets.min(), test_preds.min())
    max_val = max(test_targets.max(), test_preds.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    r2 = r2_score(test_targets, test_preds)
    ax4.set_xlabel('True (eV)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Predicted (eV)', fontsize=12, fontweight='bold')
    ax4.set_title(f'Test Results (MAE: {test_mae:.4f} eV, R²: {r2:.4f})', 
                fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_pfp.png'), dpi=200)
    plt.close()
    
    # 統計保存
    stats = {
        'dataset': 'tmQM',
        'target': 'LUMO',
        'model_version': 'egnn_with_pfp_matlantis',
        'paper_compliant': True,
        'reference': 'EGNN (Satorras et al., 2021)',
        'pfp_source': 'Matlantis (Takamoto et al., 2022)',
        'training': {
            'best_val_mae': float(best_val_mae),
            'test_mae': float(test_mae),
            'epochs_trained': epoch + 1,
            'num_parameters': num_params,
            'batch_size': batch_size,
            'initial_lr': initial_lr
        },
        'model_config': {
            'input_features': '261-dim (PFP256 + atomic_num1 + geom4)',
            'edge_features': '4-dim',
            'hidden_dim': 128,
            'num_layers': 7,
            'coord_update': True,
            'edge_inference': True,
            'pfp_enabled': True,
            'pfp_per_atom': True,
            'pooling': 'mean'
        },
        'data_split': {
            'method': 'stratified_by_metal',
            'train_size': len(train_graphs),
            'val_size': len(val_graphs),
            'test_size': len(test_graphs)
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(OUTPUT_DIR, 'training_stats_pfp.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*70}")
    print("🎉 EGNN×PFP Training Complete (tmQM LUMO)!")
    print(f"{'='*70}")
    print(f"Best Val MAE:  {best_val_mae:.4f} eV")
    print(f"Test MAE:      {test_mae:.4f} eV")
    print(f"Output:        {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
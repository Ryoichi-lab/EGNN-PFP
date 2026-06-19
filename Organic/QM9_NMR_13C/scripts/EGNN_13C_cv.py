#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QM9NMR ¹³C NMR遮蔽定数予測 - EGNN-PFP 5-Fold Cross Validation
先行研究（matlantis-contrib）に合わせた GroupKFold(5) 準拠
各 fold を逐次実行し、最後に mean ± std を集計
ノード特徴: PFP(256) + one-hot元素(5) + 距離特徴(4) = 265次元
エッジ特徴: [距離, cos_sim, L2_dist, 距離×cos_sim] = 4次元
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
from sklearn.metrics import r2_score


# =============================================
# Utility functions
# =============================================

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids_expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count  = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids_expanded, data)
    count.scatter_add_(0, segment_ids_expanded, torch.ones_like(data))
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
        self.epsilon   = 1e-8

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + 1 + edges_in_d, hidden_nf),
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
        edge_feat  = self.edge_model(h[row], h[col], radial, edge_attr)
        edge_feat  = edge_feat * edge_mask
        h, _       = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


# =============================================
# EGNN-PFP モデル
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
                act_fn=act_fn,
                recurrent=True,
                coords_weight=coords_weight,
                attention=attention
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
# バッチ前処理（¹³C: c_mask）
# =============================================

def prepare_batch(batch, device):
    batch_size = batch.batch.max().item() + 1
    node_counts, max_nodes = [], 0
    for i in range(batch_size):
        n = (batch.batch == i).sum().item()
        node_counts.append(n)
        max_nodes = max(max_nodes, n)

    total_nodes  = batch_size * max_nodes
    h0           = torch.zeros(total_nodes, batch.x.size(1), device=device)
    x            = torch.zeros(total_nodes, 3, device=device)
    node_mask    = torch.zeros(total_nodes, 1, device=device)
    target       = torch.zeros(total_nodes, device=device)
    target_mask  = torch.zeros(total_nodes, dtype=torch.bool, device=device)

    for i, n in enumerate(node_counts):
        mask = (batch.batch == i)
        s    = i * max_nodes
        h0[s:s+n]          = batch.x[mask].to(device)
        x[s:s+n]           = batch.pos[mask].to(device)
        node_mask[s:s+n]   = 1.0
        target[s:s+n]      = batch.y[mask].to(device)
        target_mask[s:s+n] = batch.c_mask[mask].to(device)

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
# Cosine Annealing Scheduler
# =============================================

class CosineAnnealingScheduler:
    def __init__(self, optimizer, max_epochs, eta_min=1e-7):
        self.optimizer  = optimizer
        self.max_epochs = max_epochs
        self.eta_min    = eta_min
        self.base_lr    = optimizer.param_groups[0]['lr']
        self.epoch      = 0

    def step(self):
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             0.5 * (1 + math.cos(math.pi * self.epoch / self.max_epochs))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.epoch += 1
        return lr


# =============================================
# 訓練・評価
# =============================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_mae, n_samples = 0, 0, 0
    for batch in tqdm(loader, desc="Train", leave=False):
        batch = batch.to(device)
        h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, target, tmask = \
            prepare_batch(batch, device)
        optimizer.zero_grad()
        pred = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
        p, t = pred[tmask], target[tmask]
        loss = nn.MSELoss()(p, t)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        mae = torch.abs(p - t).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if not torch.isnan(mae):
            total_loss += loss.item() * len(t)
            total_mae  += mae.item()  * len(t)
            n_samples  += len(t)
    return (total_loss / n_samples, total_mae / n_samples) if n_samples else (float('inf'), float('inf'))


def validate_epoch(model, loader, device):
    model.eval()
    total_mae, n_samples = 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val  ", leave=False):
            batch = batch.to(device)
            h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, target, tmask = \
                prepare_batch(batch, device)
            pred = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
            if torch.isnan(pred).any():
                continue
            p, t = pred[tmask], target[tmask]
            mae  = torch.abs(p - t).mean()
            if not torch.isnan(mae):
                total_mae  += mae.item() * len(t)
                n_samples  += len(t)
                all_preds.append(p.cpu().numpy())
                all_targets.append(t.cpu().numpy())
    if n_samples == 0:
        return float('inf'), np.array([]), np.array([])
    return (total_mae / n_samples,
            np.concatenate(all_preds),
            np.concatenate(all_targets))


def plot_predictions(preds, targets, epoch, mae, output_dir, label='13C'):
    mask = ~(np.isnan(preds) | np.isnan(targets) | np.isinf(preds) | np.isinf(targets))
    if mask.sum() == 0:
        return
    preds, targets = preds[mask], targets[mask]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(targets, preds, alpha=0.3, s=15, edgecolors='none')
    lo, hi = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2)
    r2 = r2_score(targets, preds)
    ax.set_xlabel(f'True ¹{label} NMR [ppm]', fontsize=12)
    ax.set_ylabel(f'Predicted ¹{label} NMR [ppm]', fontsize=12)
    ax.set_title(f'EGNN-PFP CV  Epoch {epoch}\nMAE={mae:.4f} ppm  R²={r2:.4f}', fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    err = preds - targets
    ax.hist(err, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', ls='--', lw=2)
    ax.set_xlabel('Error [ppm]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Error Distribution\nMean={err.mean():.4f}  Std={err.std():.4f}', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{label}_NMR_epoch_{epoch}.png'), dpi=200)
    plt.close()


# =============================================
# 1 fold 分の訓練
# =============================================

def train_one_fold(val_fold, graph_dir, output_base, device, num_epochs=500):
    train_folds = [i for i in range(5) if i != val_fold]
    fold_dir = os.path.join(output_base, f"fold_{val_fold}")
    os.makedirs(fold_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Fold {val_fold}/4  |  Train={train_folds}  Val=[{val_fold}]")
    print(f"{'='*70}")

    # データ読み込み
    train_graphs = []
    for fi in train_folds:
        train_graphs += torch.load(os.path.join(graph_dir, f"fold_{fi}_graphs.pt"), weights_only=False)
    val_graphs = torch.load(os.path.join(graph_dir, f"fold_{val_fold}_graphs.pt"), weights_only=False)

    # 外れ値除去（¹³C: -50 ppm未満）
    def filter_outliers(graphs, thr=-50.0):
        ok, n_rem = [], 0
        for g in graphs:
            if (g.y[g.c_mask.bool()] < thr).any():
                n_rem += 1
            else:
                ok.append(g)
        return ok, n_rem

    train_graphs, r_tr = filter_outliers(train_graphs)
    val_graphs,   r_va = filter_outliers(val_graphs)
    print(f"  外れ値除去 Train: {r_tr} / Val: {r_va}")
    print(f"  Train: {len(train_graphs):,} mol  |  Val: {len(val_graphs):,} mol")

    batch_size   = 32
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False, num_workers=0)

    # モデル（各 fold で新規初期化）
    model = EGNN_NMR(
        in_node_nf   = 265,
        in_edge_nf   = 4,
        hidden_nf    = 128,
        device       = device,
        act_fn       = nn.SiLU(),
        n_layers     = 7,
        coords_weight= 1.0,
        attention    = True,
        node_attr    = True
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    initial_lr = 5e-4
    optimizer  = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-8)
    scheduler  = CosineAnnealingScheduler(optimizer, num_epochs, eta_min=1e-7)

    best_val_mae = float('inf')
    patience, patience_counter, min_improve = 700, 0, 0.001
    history = {'train_loss': [], 'train_mae': [], 'val_mae': [], 'lr': []}

    print(f"\n  学習開始 (Fold {val_fold})...\n")
    for epoch in range(num_epochs):
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, device)
        val_mae, val_preds, val_targets = validate_epoch(model, val_loader, device)
        lr = scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['lr'].append(lr)

        print(f"[Fold {val_fold}] Epoch {epoch+1:3d}/{num_epochs}  "
              f"TrainLoss={train_loss:.4f}  TrainMAE={train_mae:.4f}  "
              f"ValMAE={val_mae:.4f}  LR={lr:.2e}")

        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mae': val_mae,
            'history': history
        }

        if val_mae < best_val_mae - min_improve:
            improve = best_val_mae - val_mae
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(ckpt, os.path.join(fold_dir, 'best_model.pth'))
            print(f"  ★ Best! improved {improve:.4f} ppm")
            if (epoch + 1) % 50 == 0:
                plot_predictions(val_preds, val_targets, epoch+1, val_mae, fold_dir, '13C')
        else:
            patience_counter += 1

        if (epoch + 1) % 100 == 0:
            torch.save(ckpt, os.path.join(fold_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            plot_predictions(val_preds, val_targets, epoch+1, val_mae, fold_dir, '13C')

        if patience_counter >= patience:
            print(f"\n  Early stopping (Fold {val_fold}).")
            break

    # fold の統計を保存
    fold_stats = {
        'val_fold': val_fold,
        'train_folds': train_folds,
        'best_val_mae': float(best_val_mae),
        'stopped_epoch': len(history['val_mae']),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(fold_dir, 'fold_stats.json'), 'w') as f:
        json.dump(fold_stats, f, indent=2)

    print(f"\n  [Fold {val_fold}] Best ValMAE: {best_val_mae:.4f} ppm")
    return best_val_mae


# =============================================
# メイン
# =============================================

def main():
    GRAPH_DIR   = os.path.expanduser("~/qm9nmr/EGNN_PFP/graphs/13C")
    OUTPUT_BASE = os.path.expanduser("~/qm9nmr/EGNN_PFP/training_13C_cv")
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 500
    n_folds    = 5

    print("=" * 70)
    print("QM9NMR ¹³C NMR遮蔽定数予測 - EGNN-PFP  5-Fold CV")
    print("先行研究（matlantis-contrib）準拠")
    print("=" * 70)
    print(f"Device: {device}  |  Folds: {n_folds}  |  Epochs/fold: {num_epochs}")

    fold_results = []
    for val_fold in range(n_folds):
        best_mae = train_one_fold(val_fold, GRAPH_DIR, OUTPUT_BASE, device, num_epochs)
        fold_results.append(best_mae)

    # =============================================
    # CV サマリー
    # =============================================
    mean_mae = float(np.mean(fold_results))
    std_mae  = float(np.std(fold_results))

    print(f"\n{'='*70}")
    print("  5-Fold CV 結果サマリー  ¹³C NMR PFP")
    print(f"{'='*70}")
    for i, v in enumerate(fold_results):
        print(f"  Fold {i}: Best ValMAE = {v:.4f} ppm")
    print(f"  {'─'*40}")
    print(f"  Mean ± Std = {mean_mae:.4f} ± {std_mae:.4f} ppm")
    print(f"{'='*70}")

    cv_summary = {
        'target': '13C_NMR_shielding',
        'model': 'EGNN-PFP',
        'n_folds': n_folds,
        'fold_best_val_mae': fold_results,
        'mean_val_mae': mean_mae,
        'std_val_mae': std_mae,
        'model_config': {'in_node_nf': 265, 'in_edge_nf': 4, 'hidden_nf': 128,
                         'n_layers': 7, 'attention': True},
        'training_config': {'epochs': num_epochs, 'batch_size': 32,
                            'lr': 5e-4, 'weight_decay': 1e-8,
                            'scheduler': 'CosineAnnealing', 'patience': 700},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(OUTPUT_BASE, 'cv_summary.json'), 'w') as f:
        json.dump(cv_summary, f, indent=2)
    print(f"\n出力: {OUTPUT_BASE}/cv_summary.json")


if __name__ == "__main__":
    main()

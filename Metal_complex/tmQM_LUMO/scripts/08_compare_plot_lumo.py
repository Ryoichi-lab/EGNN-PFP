#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tmQM LUMO予測 - Baseline vs PFP+BO 比較プロット
"""

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.metrics import r2_score, mean_absolute_error
# Hartree to eV conversion factor（ファイル冒頭に追加）
HARTREE_TO_EV = 27.211386245988

# =============================================
# モデル定義（Baseline用）
# =============================================

class EGNNLayerBaseline(MessagePassing):
    """EGNN Baseline Layer"""
    def __init__(self, hidden_dim, activation='swish'):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + 1, hidden_dim),
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
        return h + h_updated, pos + coord_update
    
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


class EGNN_Baseline_LUMO(nn.Module):
    """EGNN Baseline for LUMO"""
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=7):
        super().__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.egnn_layers = nn.ModuleList([
            EGNNLayerBaseline(hidden_dim=hidden_dim, activation='swish')
            for _ in range(num_layers)
        ])
        
        self.lumo_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, batch):
        h = self.node_encoder(batch.x)
        pos = batch.pos
        
        for layer in self.egnn_layers:
            h, pos = layer(h, batch.edge_index, batch.edge_attr, pos)
        
        # Mean pooling (energy-related property)
        graph_embedding = scatter(h, batch.batch, dim=0, reduce='mean')
        lumo_pred = self.lumo_head(graph_embedding)
        
        return {
            'lumo_energy': lumo_pred,
            'node_embeddings': h,
            'updated_positions': pos
        }


# =============================================
# モデル定義（PFP+BO用）
# =============================================

class EGNNLayerWithEdgeFeatures(MessagePassing):
    """EGNN with Edge Features"""
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
        return h + h_updated, pos + coord_update

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


class EGNN_PFP_BO_Lumo(nn.Module):
    """EGNN with PFP and Bond Order for LUMO"""
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
                                      edge_dim=hidden_dim // 4,
                                      activation='swish')
            for _ in range(num_layers)
        ])
        
        # 大文字の LUMO_head に変更
        self.LUMO_head = nn.Sequential(
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
        
        # Mean pooling (energy-related property)
        graph_embedding = scatter(h, batch.batch, dim=0, reduce='mean')
        lumo_pred = self.LUMO_head(graph_embedding)  # LUMO_head を使用
        
        return {
            'lumo': lumo_pred,
            'node_embeddings': h,
            'updated_positions': pos
        }

# =============================================
# 評価関数（単位変換対応版）
# =============================================

def evaluate_model(model, dataloader, device, convert_to_ev=True):
    """モデル評価（単位変換オプション付き）"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            outputs = model(batch)
            
            # Baselineは'lumo_energy', PFP+BOは'lumo'キーを使用
            if 'lumo_energy' in outputs:
                preds = outputs['lumo_energy'].view(-1).cpu().numpy()
            else:
                preds = outputs['lumo'].view(-1).cpu().numpy()
            
            targets = batch.y.view(-1).cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(targets)
    
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # デバッグ情報
    print(f"[DEBUG] Raw predictions: min={predictions.min():.6f}, max={predictions.max():.6f}, mean={predictions.mean():.6f}")
    print(f"[DEBUG] Raw targets: min={targets.min():.6f}, max={targets.max():.6f}, mean={targets.mean():.6f}")
    
    # 単位変換（Hartree → eV）
    if convert_to_ev:
        predictions = predictions * HARTREE_TO_EV
        targets = targets * HARTREE_TO_EV
        print(f"[DEBUG] After eV conversion: pred range=[{predictions.min():.4f}, {predictions.max():.4f}] eV")
        print(f"[DEBUG] After eV conversion: target range=[{targets.min():.4f}, {targets.max():.4f}] eV")
    
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return predictions, targets, mae, r2


# =============================================
# メイン処理
# =============================================

def main():
    # パス設定
    TMQM_DIR = "/home/users/uchiyama/tmQM_dipole"
    
    BASELINE_MODEL_PATH = os.path.join(TMQM_DIR, "training_baseline_egnn_lumo2/best_baseline_model.pth")
    BASELINE_GRAPH_DIR = os.path.join(TMQM_DIR, "graphs_tmQM_baseline_lumo2")
    
    PFP_BO_MODEL_PATH = os.path.join(TMQM_DIR, "training_pfp_bo_egnn_lumo2/best_model_bo.pth")
    PFP_BO_GRAPH_DIR = os.path.join(TMQM_DIR, "graphs_tmQM_pfp_lumo_bo2")
    
    OUTPUT_DIR = os.path.join(TMQM_DIR, "comparison_plots_lumo2")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # =============================================
    # Baseline モデル評価
    # =============================================
    print("\n" + "="*70)
    print("📊 Baseline Model Evaluation (LUMO)")
    print("="*70)
    
    # データ読み込み
    baseline_test = torch.load(os.path.join(BASELINE_GRAPH_DIR, "test_graphs.pt"), weights_only=False)
    baseline_loader = DataLoader(baseline_test, batch_size=32, shuffle=False)
    
    # モデル読み込み
    baseline_model = EGNN_Baseline_LUMO(input_dim=5, hidden_dim=128, num_layers=7).to(device)
    baseline_checkpoint = torch.load(BASELINE_MODEL_PATH, map_location=device)
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    # Baseline評価（単位変換あり）
    baseline_preds, baseline_targets, baseline_mae, baseline_r2 = evaluate_model(
    baseline_model, baseline_loader, device, convert_to_ev=True
    )
    
    print(f"✓ Baseline MAE: {baseline_mae:.4f} eV")
    print(f"✓ Baseline R²:  {baseline_r2:.4f}")
    # =============================================
    # PFP+BO モデル評価
    # =============================================
    print("\n" + "="*70)
    print("📊 PFP+BO Model Evaluation (LUMO)")
    print("="*70)
    
    # データ読み込み
    pfp_bo_test = torch.load(os.path.join(PFP_BO_GRAPH_DIR, "test_graphs.pt"), weights_only=False)
    pfp_bo_loader = DataLoader(pfp_bo_test, batch_size=32, shuffle=False)
    
    # モデル読み込み
    pfp_bo_model = EGNN_PFP_BO_Lumo(input_dim=261, hidden_dim=128, 
                                     edge_dim=5, num_layers=7).to(device)
    pfp_bo_checkpoint = torch.load(PFP_BO_MODEL_PATH, map_location=device)
    pfp_bo_model.load_state_dict(pfp_bo_checkpoint['model_state_dict'])
    
    pfp_bo_preds, pfp_bo_targets, pfp_bo_mae, pfp_bo_r2 = evaluate_model(
        pfp_bo_model, pfp_bo_loader, device, convert_to_ev=True
    )
    
    print(f"✓ PFP+BO MAE: {pfp_bo_mae:.4f} eV")
    print(f"✓ PFP+BO R²:  {pfp_bo_r2:.4f}")
# =============================================
    # 比較プロット（横並び・統一サイズ版）- LUMO修正版
    # =============================================
    print("\n" + "="*70)
    print("📈 Generating Comparison Plot")
    print("="*70)
    
    # 完全に統一されたサイズ設定
    fig = plt.figure(figsize=(20, 9))
    axes = fig.subplots(1, 2)
    
    # 共通の軸範囲を計算
    all_targets = np.concatenate([baseline_targets, pfp_bo_targets])
    all_preds = np.concatenate([baseline_preds, pfp_bo_preds])
    min_val = min(all_targets.min(), all_preds.min())
    max_val = max(all_targets.max(), all_preds.max())
    
    # 軸範囲を少し余裕を持たせる
    range_val = max_val - min_val
    margin = range_val * 0.02
    plot_min = min_val - margin
    plot_max = max_val + margin
    
    # Baseline プロット
    ax1 = axes[0]
    ax1.scatter(baseline_targets, baseline_preds, alpha=0.5, s=25, 
                edgecolors='none', c='coral')
    ax1.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=2.5)
    
    ax1.set_xlabel('True LUMO Energy (eV)', fontsize=24, fontweight='bold')
    ax1.set_ylabel('Predicted LUMO Energy (eV)', fontsize=24, fontweight='bold')
    ax1.set_title('Baseline', fontsize=26, fontweight='bold', pad=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.grid(True, alpha=0.3, linewidth=1.2)
    ax1.set_xlim(plot_min, plot_max)
    ax1.set_ylim(plot_min, plot_max)
    ax1.set_aspect('equal')
    
    # 枠線を追加
    for spine in ax1.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    
    # PFP+BO プロット
    ax2 = axes[1]
    ax2.scatter(pfp_bo_targets, pfp_bo_preds, alpha=0.5, s=25, 
                edgecolors='none', c='skyblue')
    ax2.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=2.5)
    
    ax2.set_xlabel('True LUMO Energy (eV)', fontsize=24, fontweight='bold')
    ax2.set_ylabel('Predicted LUMO Energy (eV)', fontsize=24, fontweight='bold')
    ax2.set_title('EGNN-PFP', fontsize=26, fontweight='bold', pad=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.grid(True, alpha=0.3, linewidth=1.2)
    ax2.set_xlim(plot_min, plot_max)
    ax2.set_ylim(plot_min, plot_max)
    ax2.set_aspect('equal')
    
    # 枠線を追加
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    
    # 保存パスの定義
    plot_path = os.path.join(OUTPUT_DIR, 'baseline_vs_pfp_bo_lumo_comparison.png')
    
    # 統一された保存設定
    plt.tight_layout(pad=2.0)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ Plot saved: {plot_path}")
    
    improvement = ((baseline_mae - pfp_bo_mae) / baseline_mae) * 100
    print("\n" + "="*70)
    print("🎉 Comparison Complete!")
    print("="*70)
    print(f"Baseline MAE:  {baseline_mae:.4f} eV  (R²: {baseline_r2:.4f})")
    print(f"PFP+BO MAE:    {pfp_bo_mae:.4f} eV  (R²: {pfp_bo_r2:.4f})")
    print(f"Improvement:   {improvement:.1f}%")
    print("="*70 + "\n")

    # =============================================
    # 比較プロット（重ね合わせ版）- 追加
    # =============================================
    print("\n" + "="*70)
    print("📈 Generating Overlay Comparison Plot")
    print("="*70)

    # 統一されたサイズ設定
    fig, ax = plt.subplots(figsize=(12, 11))

    # 共通の軸範囲を計算（既存の変数を再利用）
    # all_targets, all_preds, plot_min, plot_max は既に計算済み

    # Baseline プロット（下層・オレンジ）
    ax.scatter(baseline_targets, baseline_preds, alpha=0.6, s=30, 
            edgecolors='none', c='coral', label='EGNN')

    # PFP+BO プロット（上層・薄青色）
    ax.scatter(pfp_bo_targets, pfp_bo_preds, alpha=0.5, s=30, 
            edgecolors='none', c='skyblue', label='EGNN-PFP')

    # 対角線（凡例なし）
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=2.5)

    ax.set_xlabel('True LUMO Energy (eV)', fontsize=38, fontweight='bold')
    ax.set_ylabel('Predicted LUMO Energy (eV)', fontsize=38, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.grid(True, alpha=0.3, linewidth=1.2)
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect('equal')

    # 凡例を右下に配置（マーカーサイズを大きく、フォントサイズも拡大）
    ax.legend(loc='lower right', fontsize=32, framealpha=0.9, edgecolor='black', 
            fancybox=False, shadow=False, markerscale=2.5)

    # 枠線を追加
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    # 図のラベルを追加（左上）
    ax.text(0.05, 0.95, '(d) LUMO Energy', transform=ax.transAxes,
            fontsize=36, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5))

    # 保存パスの定義
    overlay_plot_path = os.path.join(OUTPUT_DIR, 'baseline_vs_pfp_bo_overlay.png')

    # 統一された保存設定
    plt.tight_layout(pad=1.5)
    plt.savefig(overlay_plot_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ Overlay plot saved: {overlay_plot_path}")
        

if __name__ == "__main__":
    main()
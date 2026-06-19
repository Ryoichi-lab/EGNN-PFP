#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tmQM HOMO-LUMO Gap Prediction - Baseline vs PFP Error Distribution Comparison
"""

import os
import sys
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

# モデル定義をインポート
sys.path.append('/home/users/uchiyama/tmQM_dipole/HOMO_LUMO')

# 単位変換定数
HARTREE_TO_EV = 27.2114

# =============================================
# EGNNLayerBaseline (from 3.make_EGNN_base.py)
# =============================================

class EGNNLayerBaseline(MessagePassing):
    """EGNN原著論文準拠（距離のみ）"""
    def __init__(self, hidden_dim, activation='swish'):
        super().__init__(aggr='add')
        
        self.hidden_dim = hidden_dim
        
        # φe: メッセージ関数（距離のみ）
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + 1, hidden_dim),  # h_i, h_j, dist_sq, edge_dist
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
        """論文 式(3)"""
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
# EGNN_Baseline_HomoLumo (from 3.make_EGNN_base.py)
# =============================================

class EGNN_Baseline_HomoLumo(nn.Module):
    """
    EGNN Baseline - HOMO-LUMOギャップ予測用（PFPなし）
    入力: 原子番号(1) + 幾何特徴(4) = 5次元
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
            EGNNLayerBaseline(
                hidden_dim=hidden_dim,
                activation='swish'
            )
            for _ in range(num_layers)
        ])
        
        # HOMO-LUMOギャップ予測ヘッド
        self.gap_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, batch):
        # エンコーディング
        h = self.node_encoder(batch.x)
        pos = batch.pos
        
        # EGNN層（エッジ特徴は距離のみ）
        for layer in self.egnn_layers:
            h, pos = layer(h, batch.edge_index, batch.edge_attr, pos)
        
        # グラフレベルの集約（mean pooling - エネルギー関連なので平均を使用）
        graph_embedding = scatter(h, batch.batch, dim=0, reduce='mean')
        
        # HOMO-LUMOギャップ予測
        gap_pred = self.gap_head(graph_embedding)
        
        return {
            'homolumo_gap': gap_pred,
            'node_embeddings': h,
            'updated_positions': pos
        }


# =============================================
# EGNNLayerWithEdgeFeatures (from 2.make_EGNN_pfp.py)
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
# EGNN_PFP_HomoLumo (from 2.make_EGNN_pfp.py)
# =============================================

class EGNN_PFP_HomoLumo(nn.Module):
    """
    EGNN×PFP - HOMO-LUMOギャップ予測用
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
        
        # HOMO-LUMOギャップ予測ヘッド
        self.gap_head = nn.Sequential(
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
        
        # HOMO-LUMOギャップ予測
        gap_pred = self.gap_head(graph_embedding)
        
        return {
            'homolumo_gap': gap_pred,
            'node_embeddings': h,
            'updated_positions': pos
        }


# =============================================
# 予測実行関数
# =============================================

def get_predictions(model, dataloader, device):
    """モデルで予測を実行（eV単位に変換）"""
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", leave=False):
            batch = batch.to(device)
            outputs = model(batch)
            pred = outputs['homolumo_gap'].view(-1)
            true = batch.y.view(-1)
            # Hartree から eV に変換
            all_preds.extend((pred * HARTREE_TO_EV).cpu().numpy())
            all_trues.extend((true * HARTREE_TO_EV).cpu().numpy())
    
    return np.array(all_preds), np.array(all_trues)


def plot_pfp_contribution_yy(model_data, output_dir):
    """PFP寄与のみのy-yプロット（eV単位）"""
    if "EGNN×PFP" not in model_data:
        print("⚠️  EGNN×PFPモデルが見つかりません。スキップします。")
        return
    
    pfp_data = model_data["EGNN×PFP"]
    
    plt.figure(figsize=(10, 10))
        

    min_val = pfp_data["trues"].min()
    max_val = pfp_data["trues"].max()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2.5, label='Perfect Prediction')
    
    plt.xlabel("True HOMO-LUMO Gap (eV)", fontsize=18, fontweight='bold')
    plt.ylabel("PFP-Predicted HOMO-LUMO Gap (eV)", fontsize=18, fontweight='bold')
    plt.title("PFP Contribution Only - HOMO-LUMO Gap (tmQM)", fontsize=19, fontweight='bold')
    plt.legend(frameon=True, loc='upper left', fontsize=16)
    plt.grid(alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "pfp_only_homolumo_yy_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ PFP寄与y-yプロットを保存: {output_path}")


# =============================================
# メイン処理
# =============================================

def main():
    TMQM_DIR = "/home/users/uchiyama/tmQM_dipole"
    
    # PFP用とBaseline用で異なるグラフディレクトリ
    GRAPH_DIR_PFP = os.path.join(TMQM_DIR, "graphs_tmQM_pfp_homolumo")
    GRAPH_DIR_BASELINE = os.path.join(TMQM_DIR, "graphs_tmQM_baseline_homolumo")
    
    OUTPUT_DIR = os.path.join(TMQM_DIR, "comparison_figs_homolumo")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # モデル設定
    model_configs = {
        "Baseline": {
            "checkpoint": os.path.join(TMQM_DIR, "training_baseline_egnn_homolumo/best_baseline_model.pth"),
            "model_class": EGNN_Baseline_HomoLumo,
            "graph_dir": GRAPH_DIR_BASELINE,
            "color": "#D4A574"  # 濃いベージュ
        },
        "EGNN×PFP": {
            "checkpoint": os.path.join(TMQM_DIR, "training_pfp_egnn_homolumo/best_pfp_model.pth"),
            "model_class": EGNN_PFP_HomoLumo,
            "graph_dir": GRAPH_DIR_PFP,
            "color": "#5B8DB8"  # 濃い青
        }
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 各モデルで予測実行
    model_data = {}

    for model_name, config in model_configs.items():
        print(f"\n{model_name} モデル読み込み中...")
        
        # それぞれのグラフディレクトリからテストデータを読み込む
        print(f"  グラフディレクトリ: {config['graph_dir']}")
        test_graphs = torch.load(
            os.path.join(config["graph_dir"], "test_graphs.pt"), 
            map_location='cpu',
            weights_only=False
        )
        test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False, num_workers=4)
        print(f"✓ Test: {len(test_graphs)} graphs")
        
        # モデルインスタンス作成
        model = config["model_class"]().to(device)
        
        # チェックポイント読み込み
        checkpoint = torch.load(config["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ エポック {checkpoint.get('epoch', 'N/A')}, Val MAE: {checkpoint.get('val_mae', 'N/A'):.4f} eV")
        
        # 予測実行
        preds, trues = get_predictions(model, test_loader, device)
        
        # 誤差計算
        errors = np.abs(preds - trues)
        mae = mean_absolute_error(trues, preds)
        r2 = r2_score(trues, preds)
        
        model_data[model_name] = {
            "errors": errors,
            "preds": preds,
            "trues": trues,
            "mae": mae,
            "r2": r2,
            "color": config["color"]
        }
        
        print(f"✓ MAE: {mae:.4f} eV, R²: {r2:.4f}")

    # Absolute Error分布プロット（eV単位）
    print("\n誤差分布プロット作成中...")
    plt.figure(figsize=(12, 7))

    all_errors = np.concatenate([data["errors"] for data in model_data.values()])
    bins = np.linspace(0, np.percentile(all_errors, 99.5), 50)

    for model_name, data in model_data.items():
        plt.hist(
            data["errors"],
            bins=bins,
            alpha=0.75,
            color=data["color"],
            label=f'{model_name}\nMAE = {data["mae"]:.4f} eV',
            edgecolor='black',
            linewidth=0.5
        )
        
        plt.axvline(
            data["mae"],
            color=data["color"],
            linestyle='--',
            lw=2.5,
            alpha=0.9
        )

    plt.xlim(0, np.percentile(all_errors, 99.5))
    plt.grid(alpha=0.3)
    plt.xlabel("Absolute Error (eV)", fontsize=18, fontweight='bold')
    plt.ylabel("Frequency", fontsize=18, fontweight='bold')
    plt.title("HOMO-LUMO Gap - Error Distribution Comparison (tmQM)", fontsize=19, fontweight='bold')
    plt.legend(frameon=True, loc='upper right', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "baseline_vs_pfp_homolumo_error_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 誤差分布図を保存: {output_path}")

    # y-yプロット（eV単位）
    print("\ny-yプロット作成中...")
    plt.figure(figsize=(10, 10))

    for model_name, data in model_data.items():
        plt.scatter(
            data["trues"],
            data["preds"],
            s=8,
            alpha=0.4,
            color=data["color"],
            label=f'{model_name} (R²={data["r2"]:.4f})'
        )

    min_val = min([data["trues"].min() for data in model_data.values()])
    max_val = max([data["trues"].max() for data in model_data.values()])
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2.5, label='Perfect Prediction')

    plt.xlabel("True HOMO-LUMO Gap (eV)", fontsize=18, fontweight='bold')
    plt.ylabel("Predicted HOMO-LUMO Gap (eV)", fontsize=18, fontweight='bold')
    plt.title("HOMO-LUMO Gap - Baseline vs EGNN×PFP (tmQM)", fontsize=19, fontweight='bold')
    plt.legend(frameon=True, loc='upper left', fontsize=18)
    plt.grid(alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()

    output_path_yy = os.path.join(OUTPUT_DIR, "baseline_vs_pfp_homolumo_yy_plot.png")
    plt.savefig(output_path_yy, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ y-yプロットを保存: {output_path_yy}")
    
    # PFP寄与のみのy-yプロット
    print("\nPFP寄与のみのy-yプロット作成中...")
    plot_pfp_contribution_yy(model_data, OUTPUT_DIR)

    # 結果サマリー（eV単位）
    print("\n🎉 完了!")
    print(f"\n{'='*60}")
    print("結果サマリー")
    print(f"{'='*60}")
    for model_name, data in model_data.items():
        print(f"{model_name}: MAE = {data['mae']:.4f} eV, R² = {data['r2']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
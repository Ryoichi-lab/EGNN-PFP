#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Case Study 外挿性評価 - CV版（5-Fold アンサンブル）
training_13C_cv/fold_{i}/best_model.pth と
training_1H_cv/fold_{i}/best_model.pth (i=0〜4) の5モデルをアンサンブル
5カテゴリ (12drugs, 40drugs, GDB, PAH, pyrimidinone) で ¹³C / ¹H MAE を評価
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
import os, json
from datetime import datetime
from sklearn.metrics import r2_score


# =============================================
# パス設定
# =============================================

BASE_DIR       = os.path.expanduser("~/qm9nmr")
CASE_DIR       = os.path.join(BASE_DIR, "case_studies")
XYZ_DIR        = os.path.join(CASE_DIR, "split_xyz")
CV_DIR_13C      = os.path.join(BASE_DIR, "EGNN_PFP/training_13C_cv")
CV_DIR_1H       = os.path.join(BASE_DIR, "EGNN_PFP/training_1H_cv")
OUTPUT_DIR      = os.path.join(BASE_DIR, "EGNN_PFP/case_study_results_cv")
GRAPH_CACHE_DIR = os.path.join(BASE_DIR, "EGNN_PFP/graphs/case_studies")
N_FOLDS         = 5

CATEGORIES = {
    '12drugs':      'SI_12Drugs_DFT_NMR.txt',
    '40drugs':      'SI_40Drugs_DFT_NMR.txt',
    'GDB':          'SI_GDB10to17_DFT_NMR.txt',
    'PAH':          'SI_PAH_DFT_NMR.txt',
    'pyrimidinone': 'SI_pyrimidinone_DFT_NMR.txt',
}

ELEM_TO_IDX = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
N_ELEM      = 5


# =============================================
# データパース
# =============================================

def parse_xyz(path):
    """xyz ファイルを読んで (atoms, positions) を返す"""
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    atoms, positions = [], []
    for line in lines[2:2+n]:
        parts = line.split()
        atoms.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.array(positions, dtype=np.float32)


def parse_nmr_file(path):
    """SI_*_DFT_NMR.txt を読んで {mol_name: {atoms, shielding}} を返す"""
    data = {}
    with open(path) as f:
        lines = [l.strip() for l in f.readlines()]
    i = 0
    while i < len(lines):
        if not lines[i]:
            i += 1; continue
        if lines[i].isdigit():
            n = int(lines[i])
            mol_name = lines[i+1].strip()
            atoms, shielding = [], []
            for j in range(i+2, i+2+n):
                parts = lines[j].split()
                atoms.append(parts[0])
                shielding.append(float(parts[1]))
            data[mol_name] = {'atoms': atoms, 'shielding': shielding}
            i += 2 + n
        else:
            i += 1
    return data


# =============================================
# グラフ構築（make_graph と同一ロジック）
# =============================================

def build_node_features(positions, atoms, pfp_desc):
    center = positions.mean(axis=0)
    feats  = []
    for i in range(len(positions)):
        dists = [np.linalg.norm(positions[i] - positions[j])
                 for j in range(len(positions)) if j != i]
        oh = np.zeros(N_ELEM, dtype=np.float32)
        oh[ELEM_TO_IDX.get(atoms[i], 0)] = 1.0
        feat = np.concatenate([
            pfp_desc[i],
            oh,
            [np.linalg.norm(positions[i] - center),
             np.min(dists) if dists else 0.0,
             np.mean(dists) if dists else 0.0,
             float(np.sum(np.array(dists) < 3.0))]
        ])
        feats.append(feat)
    return np.array(feats, dtype=np.float32)


def build_edges_fully_connected(positions, pfp_desc):
    edge_index, edge_feats = [], []
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            dist    = float(np.linalg.norm(positions[i] - positions[j]))
            dot     = float(np.dot(pfp_desc[i], pfp_desc[j]))
            norm_i  = float(np.linalg.norm(pfp_desc[i]))
            norm_j  = float(np.linalg.norm(pfp_desc[j]))
            cos_sim = dot / (norm_i * norm_j + 1e-8)
            l2_dist = float(np.linalg.norm(pfp_desc[i] - pfp_desc[j]))
            feat = [dist, cos_sim, l2_dist, dist * cos_sim]
            edge_index.extend([[i, j], [j, i]])
            edge_feats.extend([feat, feat])
    return edge_index, edge_feats


def build_graph(mol_name, xyz_dir, category, pfp_data, nmr_data):
    """1分子のグラフを作成（c_mask と h_mask 両方付き）"""
    xyz_key  = mol_name + '.xyz'
    xyz_path = os.path.join(xyz_dir, category, xyz_key)
    if not os.path.exists(xyz_path):
        return None
    if xyz_key not in pfp_data:
        return None
    if mol_name not in nmr_data:
        return None

    atoms, positions = parse_xyz(xyz_path)
    pfp_desc         = pfp_data[xyz_key].astype(np.float32)
    nmr_info         = nmr_data[mol_name]

    if len(atoms) != pfp_desc.shape[0]:
        return None
    if len(atoms) != len(nmr_info['atoms']):
        return None

    node_feats          = build_node_features(positions, atoms, pfp_desc)
    edge_index_list, edge_feats = build_edges_fully_connected(positions, pfp_desc)

    y         = torch.tensor(nmr_info['shielding'], dtype=torch.float32)
    c_mask    = torch.tensor([a == 'C' for a in atoms], dtype=torch.float32)
    h_mask    = torch.tensor([a == 'H' for a in atoms], dtype=torch.float32)
    edge_idx  = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous() \
                if edge_index_list else torch.zeros(2, 0, dtype=torch.long)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32) \
                if edge_feats else torch.zeros(0, 4)

    return Data(
        x          = torch.tensor(node_feats, dtype=torch.float32),
        pos        = torch.tensor(positions,  dtype=torch.float32),
        edge_index = edge_idx,
        edge_attr  = edge_attr,
        y          = y,
        c_mask     = c_mask,
        h_mask     = h_mask,
        mol_name   = mol_name,
    )


# =============================================
# EGNN モデル定義（訓練スクリプトと同一）
# =============================================

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    seg_exp = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result  = data.new_full(result_shape, 0)
    count   = data.new_full(result_shape, 0)
    result.scatter_add_(0, seg_exp, data)
    count.scatter_add_(0, seg_exp, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL_mask(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
                 nodes_attr_dim=0, act_fn=nn.SiLU(), recurrent=True,
                 coords_weight=1.0, attention=False):
        super().__init__()
        self.recurrent = recurrent
        self.attention = attention
        self.edge_mlp  = nn.Sequential(
            nn.Linear(input_nf * 2 + 1 + edges_in_d, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf), act_fn)
        self.node_mlp  = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_attr_dim, hidden_nf), act_fn,
            nn.Linear(hidden_nf, output_nf))
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
        radial = torch.sum(diff**2, dim=1, keepdim=True)
        return radial, diff

    def forward(self, h, edge_index, coord, node_mask, edge_mask,
                edge_attr=None, node_attr=None, n_nodes=None):
        row, col  = edge_index
        radial, _ = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        edge_feat = edge_feat * edge_mask
        h, _      = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


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
                edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr,
                act_fn=act_fn, recurrent=True,
                coords_weight=coords_weight, attention=attention))
        self.node_dec = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf))
        self.nmr_head = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn,
            nn.Linear(hidden_nf, 1))
        self.to(device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(self.n_layers):
            node_attr_in = h0 if self.node_attr else None
            h, _, _ = self._modules[f"gcl_{i}"](
                h, edges, x, node_mask, edge_mask,
                edge_attr=edge_attr, node_attr=node_attr_in, n_nodes=n_nodes)
        h    = self.node_dec(h) * node_mask
        pred = self.nmr_head(h)
        return pred.squeeze(-1)


# =============================================
# バッチ前処理（mask_attr で ¹³C / ¹H を切替）
# =============================================

def prepare_batch(batch, device, mask_attr='c_mask'):
    batch_size = batch.batch.max().item() + 1
    node_counts, max_nodes = [], 0
    for i in range(batch_size):
        n = (batch.batch == i).sum().item()
        node_counts.append(n)
        max_nodes = max(max_nodes, n)

    total_nodes = batch_size * max_nodes
    h0          = torch.zeros(total_nodes, batch.x.size(1), device=device)
    x           = torch.zeros(total_nodes, 3, device=device)
    node_mask   = torch.zeros(total_nodes, 1, device=device)
    target      = torch.zeros(total_nodes, device=device)
    target_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)

    for i, n in enumerate(node_counts):
        mask = (batch.batch == i)
        s    = i * max_nodes
        h0[s:s+n]          = batch.x[mask].to(device)
        x[s:s+n]           = batch.pos[mask].to(device)
        node_mask[s:s+n]   = 1.0
        target[s:s+n]      = batch.y[mask].to(device)
        target_mask[s:s+n] = getattr(batch, mask_attr)[mask].bool().to(device)

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
# 推論（単一モデル）
# =============================================

def run_inference_single(model, graphs, device, mask_attr, batch_size=16):
    """1モデルで推論し、atom-indexごとの予測値 dict を返す"""
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_preds, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, target, tmask = \
                prepare_batch(batch, device, mask_attr)
            pred = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
            if torch.isnan(pred).any():
                # NaN パッチ: NaN を 0 で埋めて後段で除外
                pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
            all_preds.append(pred[tmask].cpu().numpy())
            all_targets.append(target[tmask].cpu().numpy())
    if not all_preds:
        return np.array([]), np.array([])
    return np.concatenate(all_preds), np.concatenate(all_targets)


# =============================================
# アンサンブル推論（5モデル平均）
# =============================================

def run_ensemble_inference(models, graphs, device, mask_attr, batch_size=16):
    """
    models: list of EGNN_NMR (5 fold models)
    各モデルの予測を平均してアンサンブル予測を返す
    """
    fold_preds = []
    for fold_idx, model in enumerate(models):
        preds, targets = run_inference_single(model, graphs, device, mask_attr, batch_size)
        if len(preds) == 0:
            print(f"    Fold {fold_idx}: 推論結果なし（スキップ）")
            continue
        fold_preds.append(preds)
        # targets は全 fold で同じはずなので最後のものを使用

    if not fold_preds:
        return np.array([]), np.array([])

    # shape: (n_valid_folds, n_atoms)
    stacked = np.stack(fold_preds, axis=0)
    ensemble_preds = stacked.mean(axis=0)
    ensemble_std   = stacked.std(axis=0)
    return ensemble_preds, targets, ensemble_std


# =============================================
# プロット
# =============================================

def plot_scatter(preds, targets, mae, r2, title, save_path, std=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    if std is not None:
        sc = ax.scatter(targets, preds, c=std, cmap='viridis', alpha=0.5, s=20, edgecolors='none')
        plt.colorbar(sc, ax=ax, label='Ensemble Std [ppm]')
    else:
        ax.scatter(targets, preds, alpha=0.5, s=20, edgecolors='none')
    lo = min(targets.min(), preds.min()); hi = max(targets.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2)
    ax.set_xlabel('True NMR Shielding [ppm]', fontsize=12)
    ax.set_ylabel('Predicted NMR Shielding [ppm]', fontsize=12)
    ax.set_title(f'{title}\nMAE={mae:.4f} ppm  R²={r2:.4f}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    err = preds - targets
    ax.hist(err, bins=40, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', ls='--', lw=2)
    ax.set_xlabel('Error [ppm]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Error Distribution\nMean={err.mean():.4f}  Std={err.std():.4f}', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =============================================
# メイン
# =============================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"N_FOLDS: {N_FOLDS}")

    # ── モデルロード (5-fold ensemble) ──
    def load_fold_models(cv_dir, in_node_nf, in_edge_nf, nucleus):
        models = []
        for fold_i in range(N_FOLDS):
            ckpt_path = os.path.join(cv_dir, f"fold_{fold_i}", "best_model.pth")
            if not os.path.exists(ckpt_path):
                print(f"  [{nucleus}] fold_{fold_i}: チェックポイントなし ({ckpt_path})")
                continue
            model = EGNN_NMR(in_node_nf=in_node_nf, in_edge_nf=in_edge_nf,
                             hidden_nf=128, device=device, act_fn=nn.SiLU(),
                             n_layers=7, attention=True, node_attr=True).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            print(f"  [{nucleus}] fold_{fold_i}: "
                  f"epoch={ckpt.get('epoch','?')}  "
                  f"val_mae={ckpt.get('val_mae', float('nan')):.4f} ppm")
            models.append(model)
        return models

    print("\n[1] モデルロード（5-Fold Ensemble）...")
    models_13c = load_fold_models(CV_DIR_13C, in_node_nf=265, in_edge_nf=4, nucleus='13C')
    models_1h  = load_fold_models(CV_DIR_1H,  in_node_nf=265, in_edge_nf=4, nucleus='1H')

    if not models_13c:
        print("WARNING: ¹³C モデルが1つも見つかりません。評価をスキップします。")
    if not models_1h:
        print("WARNING: ¹H モデルが1つも見つかりません。評価をスキップします。")

    all_results = {
        'n_folds_13C': len(models_13c),
        'n_folds_1H':  len(models_1h),
    }

    for nucleus, models, mask_attr, outlier_thr in [
        ('13C', models_13c, 'c_mask', -50.0),
        ('1H',  models_1h,  'h_mask',   0.0),
    ]:
        if not models:
            continue

        print(f"\n{'='*70}")
        print(f"  ¹{nucleus} NMR 評価  ({len(models)} モデルアンサンブル)")
        print(f"{'='*70}")
        nucleus_results = {}
        all_cat_preds, all_cat_targets, all_cat_stds = [], [], []

        for cat, nmr_fname in CATEGORIES.items():
            # ── グラフ取得（キャッシュ優先） ──
            cache_path = os.path.join(GRAPH_CACHE_DIR, f"{cat}_graphs.pt")
            if os.path.exists(cache_path):
                print(f"  [{cat}] キャッシュをロード: {cache_path}")
                all_graphs = torch.load(cache_path, map_location='cpu', weights_only=False)
            else:
                # キャッシュなし → オンザフライ構築
                print(f"  [{cat}] キャッシュなし。オンザフライ構築中...")
                npz_path = os.path.join(CASE_DIR, f"{cat}_NMR_PFP_descriptor.npz")
                if not os.path.exists(npz_path):
                    print(f"  [{cat}] PFP npz not found: {npz_path}"); continue
                pfp_data = np.load(npz_path, allow_pickle=True)
                nmr_path = os.path.join(CASE_DIR, cat, nmr_fname)
                if not os.path.exists(nmr_path):
                    print(f"  [{cat}] NMR file not found: {nmr_path}"); continue
                nmr_data = parse_nmr_file(nmr_path)
                all_graphs = []
                for mol_name in nmr_data.keys():
                    g = build_graph(mol_name, XYZ_DIR, cat, pfp_data, nmr_data)
                    if g is not None:
                        all_graphs.append(g)

            # 外れ値除去・ターゲット原子チェック
            graphs = []
            for g in all_graphs:
                mask = getattr(g, mask_attr).bool()
                if mask.sum() == 0:
                    continue
                if (g.y[mask] < outlier_thr).any():
                    continue
                graphs.append(g)

            if not graphs:
                print(f"  [{cat}] グラフなし"); continue

            # アンサンブル推論
            result = run_ensemble_inference(models, graphs, device, mask_attr)
            if len(result[0]) == 0:
                print(f"  [{cat}] 推論結果なし"); continue
            preds, targets, std = result

            mae = float(np.mean(np.abs(preds - targets)))
            r2  = float(r2_score(targets, preds))
            mean_std = float(np.mean(std))

            nucleus_results[cat] = {
                'mae': mae,
                'r2': r2,
                'n_atoms': len(preds),
                'n_mols': len(graphs),
                'mean_ensemble_std': mean_std,
            }
            all_cat_preds.append(preds)
            all_cat_targets.append(targets)
            all_cat_stds.append(std)

            print(f"  [{cat}] {len(graphs):3d} mol  {len(preds):6d} atoms  "
                  f"MAE={mae:.4f} ppm  R²={r2:.4f}  "
                  f"EnsStd={mean_std:.4f}")

            # プロット
            cat_dir = os.path.join(OUTPUT_DIR, nucleus)
            os.makedirs(cat_dir, exist_ok=True)
            plot_scatter(preds, targets, mae, r2,
                         f'¹{nucleus} NMR  {cat}  (CV Ensemble {len(models)}-Fold)',
                         os.path.join(cat_dir, f'{cat}.png'),
                         std=std)

        # 全カテゴリまとめ
        if all_cat_preds:
            all_p = np.concatenate(all_cat_preds)
            all_t = np.concatenate(all_cat_targets)
            all_s = np.concatenate(all_cat_stds)
            overall_mae = float(np.mean(np.abs(all_p - all_t)))
            overall_r2  = float(r2_score(all_t, all_p))
            nucleus_results['overall'] = {
                'mae': overall_mae,
                'r2': overall_r2,
                'n_atoms': len(all_p),
                'mean_ensemble_std': float(np.mean(all_s)),
            }
            print(f"\n  [Overall] MAE={overall_mae:.4f} ppm  R²={overall_r2:.4f}  "
                  f"({len(all_p)} atoms)")
            plot_scatter(all_p, all_t, overall_mae, overall_r2,
                         f'¹{nucleus} NMR  All Categories  (CV Ensemble {len(models)}-Fold)',
                         os.path.join(OUTPUT_DIR, nucleus, 'overall.png'),
                         std=all_s)

        all_results[nucleus] = nucleus_results

    # JSON保存
    all_results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    all_results['version']   = 'cv_ensemble'
    out_json = os.path.join(OUTPUT_DIR, 'results_cv.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    # サマリー表示
    print(f"\n{'='*70}")
    print("  結果サマリー（CV 5-Fold アンサンブル）")
    print(f"{'='*70}")
    for nucleus in ['13C', '1H']:
        if nucleus not in all_results:
            continue
        print(f"\n  ¹{nucleus} NMR:")
        for cat, res in all_results[nucleus].items():
            if isinstance(res, dict):
                print(f"    {cat:15s}  MAE={res['mae']:.4f} ppm  "
                      f"R²={res['r2']:.4f}  "
                      f"EnsStd={res.get('mean_ensemble_std', float('nan')):.4f}")
    print(f"\n  出力: {OUTPUT_DIR}")
    print(f"  JSON: {out_json}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

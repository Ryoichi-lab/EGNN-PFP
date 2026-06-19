#!/usr/bin/env python3
"""Individual y-y parity plots for every NMR shielding model. One PNG per model."""

import torch, json
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
import os

SPLITS_DIR = '/home/users/uchiyama/relativistic_effect/finetune'
OUT_DIR    = '/home/users/uchiyama/relativistic_effect/finetune/yy_plots'
os.makedirs(OUT_DIR, exist_ok=True)

N_METALS = 18; METAL_EMB_DIM = 32; ORCA_DESC_DIM = 41

TARGET_LABEL = {
    'nonrel': r'$\sigma_\mathrm{nonrel}$ (ppm)',
    'rel':    r'$\sigma_\mathrm{rel}$ (ppm)',
    'delta':  r'$\Delta\sigma$ (ppm)',
}
FIELD_MAP = {
    'nonrel': ('y_nmr_nonrel', 'y_nmr_nonrel'),
    'rel':    ('y_nmr_rel',    'y_nmr_rel'),
    'delta':  ('y_delta_nmr',  'y_delta_nmr'),
}


# ── Layers ────────────────────────────────────────────────────────────────────
class EGNNLayer(MessagePassing):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__(aggr='add')
        self.message_mlp = nn.Sequential(nn.Linear(hidden_dim*2+1+edge_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim),nn.SiLU())
        self.coord_mlp   = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1),nn.Tanh())
        self.edge_inf    = nn.Sequential(nn.Linear(hidden_dim,1),nn.Sigmoid())
        self.node_mlp    = nn.Sequential(nn.Linear(hidden_dim*2,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim))
    def forward(self,h,ei,ea,pos):
        hu,cu=self.propagate(ei,h=h,edge_attr=ea,pos=pos); return h+hu,pos+cu
    def message(self,h_i,h_j,edge_attr,pos_i,pos_j):
        d2=torch.sum((pos_i-pos_j)**2,dim=-1,keepdim=True); m=self.message_mlp(torch.cat([h_i,h_j,d2,edge_attr],dim=-1)); return m*self.edge_inf(m)
    def propagate(self,ei,h,edge_attr,pos):
        out=super().propagate(ei,h=h,edge_attr=edge_attr,pos=pos); row,col=ei; rel=pos[row]-pos[col]; d2=torch.sum(rel**2,dim=-1,keepdim=True)
        m=self.message_mlp(torch.cat([h[row],h[col],d2,edge_attr],dim=-1)); cu=scatter(rel*self.coord_mlp(m),row,dim=0,dim_size=pos.size(0),reduce='add'); cnt=scatter(torch.ones(ei.size(1),1,device=pos.device),row,dim=0,dim_size=pos.size(0),reduce='add'); return out,cu/(cnt+1e-8)
    def aggregate(self,inputs,index,ptr=None,dim_size=None):
        out=scatter(inputs,index,dim=0,dim_size=dim_size,reduce='add'); cnt=scatter(torch.ones(inputs.size(0),1,device=inputs.device),index,dim=0,dim_size=dim_size,reduce='add'); return out/(cnt+1e-8)
    def update(self,aggr_out,h): return self.node_mlp(torch.cat([h,aggr_out],dim=-1))


class EGNN_NMR(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, edge_dim=1):
        super().__init__()
        self.node_enc = nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim))
        self.edge_enc = nn.Sequential(nn.Linear(edge_dim,hidden_dim//4),nn.SiLU(),nn.Linear(hidden_dim//4,hidden_dim//4))
        self.layers   = nn.ModuleList([EGNNLayer(hidden_dim,hidden_dim//4) for _ in range(7)])
        self.metal_emb = nn.Embedding(N_METALS, METAL_EMB_DIM)
        self.head = nn.Sequential(nn.Linear(hidden_dim+METAL_EMB_DIM,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1))
    def forward(self,batch):
        h=self.node_enc(batch.x); ef=self.edge_enc(batch.edge_attr); pos=batch.pos
        for l in self.layers: h,pos=l(h,batch.edge_index,ef,pos)
        return self.head(torch.cat([h[batch.metal_mask],self.metal_emb(batch.metal_idx)],dim=-1)).view(-1)


class EGNN_NMR_ORCA(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, edge_dim=1):
        super().__init__()
        self.node_enc = nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim))
        self.edge_enc = nn.Sequential(nn.Linear(edge_dim,hidden_dim//4),nn.SiLU(),nn.Linear(hidden_dim//4,hidden_dim//4))
        self.layers   = nn.ModuleList([EGNNLayer(hidden_dim,hidden_dim//4) for _ in range(7)])
        self.metal_emb = nn.Embedding(N_METALS, METAL_EMB_DIM)
        self.head = nn.Sequential(nn.Linear(hidden_dim+METAL_EMB_DIM+ORCA_DESC_DIM,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1))
    def forward(self,batch):
        h=self.node_enc(batch.x); ef=self.edge_enc(batch.edge_attr); pos=batch.pos
        for l in self.layers: h,pos=l(h,batch.edge_index,ef,pos)
        orca=batch.orca_desc.view(h[batch.metal_mask].size(0),-1)
        return self.head(torch.cat([h[batch.metal_mask],self.metal_emb(batch.metal_idx),orca],dim=-1)).view(-1)


@torch.no_grad()
def get_preds(model, loader, mu, sigma, field, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        preds.append((model(batch) * sigma + mu).cpu().numpy())
        trues.append(getattr(batch, field).view(-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


def r2(pred, true):
    return 1 - np.sum((pred-true)**2) / (np.sum((true-true.mean())**2) + 1e-10)


def save_yy(pred, true, title, xlabel, out_path, mae, r2v, n):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo = min(true.min(), pred.min()); hi = max(true.max(), pred.max())
    margin = (hi - lo) * 0.04
    ax.scatter(true, pred, s=10, alpha=0.5, linewidths=0, color='steelblue')
    ax.plot([lo-margin, hi+margin], [lo-margin, hi+margin], 'k--', lw=1.2)
    ax.set_xlim(lo-margin, hi+margin); ax.set_ylim(lo-margin, hi+margin)
    ax.set_xlabel(f'Calculated {xlabel}', fontsize=12)
    ax.set_ylabel(f'Predicted {xlabel}', fontsize=12)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.text(0.05, 0.93, f'MAE = {mae:.1f} ppm\nR² = {r2v:.4f}\nN = {n}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {os.path.basename(out_path)}  MAE={mae:.1f}  R²={r2v:.4f}')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(f'{SPLITS_DIR}/nmr_shielding_stats.json') as f:
    stats = json.load(f)
with open(f'{SPLITS_DIR}/nmr_orca_shielding_stats.json') as f:
    orca_stats = json.load(f)

# ── Non-ORCA single-target models ─────────────────────────────────────────────
print('=== Non-ORCA models ===')
for arch, target in [('egnn','nonrel'),('egnn','rel'),('egnn','delta'),
                     ('pfp','nonrel'),('pfp','rel'),('pfp','delta')]:
    field, skey = FIELD_MAP[target]
    mu, sigma   = stats[skey]['mean'], stats[skey]['std']
    input_dim, edge_dim = (5,1) if arch=='egnn' else (261,5)
    prefix = 'nmr_base' if arch=='egnn' else 'nmr_pfp'
    ckpt_path = f'/home/users/uchiyama/relativistic_effect/finetune_nmr_{arch}_{target}/best_model.pth'

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EGNN_NMR(input_dim=input_dim, edge_dim=edge_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    gs     = torch.load(f'{SPLITS_DIR}/{prefix}_test_graphs.pt', map_location='cpu', weights_only=False)
    loader = DataLoader(gs, batch_size=64, shuffle=False, num_workers=2)
    pred, true = get_preds(model, loader, mu, sigma, field, device)

    arch_label = 'EGNN (no PFP)' if arch=='egnn' else 'EGNN×PFP'
    title  = f'{arch_label} — {TARGET_LABEL[target]}'
    out    = f'{OUT_DIR}/nmr_yy_{arch}_{target}.png'
    save_yy(pred, true, title, TARGET_LABEL[target], out,
            float(np.abs(pred-true).mean()), r2(pred,true), len(true))

# ── ORCA models ────────────────────────────────────────────────────────────────
print('=== ORCA models ===')
for arch, target in [('egnn','nonrel'),('egnn','rel'),('egnn','delta'),
                     ('pfp','nonrel'),('pfp','rel'),('pfp','delta')]:
    field, skey = FIELD_MAP[target]
    mu, sigma   = orca_stats[skey]['mean'], orca_stats[skey]['std']
    input_dim, edge_dim = (5,1) if arch=='egnn' else (261,5)
    prefix = 'nmr_orca_base' if arch=='egnn' else 'nmr_orca_pfp'
    ckpt_path = f'/home/users/uchiyama/relativistic_effect/finetune_nmr_orca_{arch}_{target}/best_model.pth'

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EGNN_NMR_ORCA(input_dim=input_dim, edge_dim=edge_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    gs     = torch.load(f'{SPLITS_DIR}/{prefix}_test_graphs.pt', map_location='cpu', weights_only=False)
    loader = DataLoader(gs, batch_size=64, shuffle=False, num_workers=2)
    pred, true = get_preds(model, loader, mu, sigma, field, device)

    arch_label = 'EGNN (no PFP)+ORCA' if arch=='egnn' else 'EGNN×PFP+ORCA'
    title  = f'{arch_label} — {TARGET_LABEL[target]}'
    out    = f'{OUT_DIR}/nmr_yy_orca_{arch}_{target}.png'
    save_yy(pred, true, title, TARGET_LABEL[target], out,
            float(np.abs(pred-true).mean()), r2(pred,true), len(true))

print(f'\nAll plots saved to: {OUT_DIR}/')

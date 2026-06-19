#!/usr/bin/env python3
"""
Single-target NMR shielding model.
Usage:
  python train_nmr_target.py --arch egnn --target rel
  python train_nmr_target.py --arch egnn --target nonrel
  python train_nmr_target.py --arch egnn --target delta
  python train_nmr_target.py --arch pfp  --target rel
  python train_nmr_target.py --arch pfp  --target nonrel
  python train_nmr_target.py --arch pfp  --target delta
"""

import os, json, argparse, torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing

N_METALS      = 18
METAL_EMB_DIM = 32
SPLITS_DIR    = "/home/users/uchiyama/relativistic_effect/finetune"


# ── Model ──────────────────────────────────────────────────────────────────────
class EGNNLayer(MessagePassing):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__(aggr='add')
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2+1+edge_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1), nn.Tanh()
        )
        self.edge_inf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, edge_index, edge_attr, pos):
        hu, cu = self.propagate(edge_index, h=h, edge_attr=edge_attr, pos=pos)
        return h + hu, pos + cu

    def message(self, h_i, h_j, edge_attr, pos_i, pos_j):
        d2 = torch.sum((pos_i - pos_j)**2, dim=-1, keepdim=True)
        m  = self.message_mlp(torch.cat([h_i, h_j, d2, edge_attr], dim=-1))
        return m * self.edge_inf(m)

    def propagate(self, edge_index, h, edge_attr, pos):
        out = super().propagate(edge_index, h=h, edge_attr=edge_attr, pos=pos)
        row, col = edge_index
        rel = pos[row] - pos[col]
        d2  = torch.sum(rel**2, dim=-1, keepdim=True)
        m   = self.message_mlp(torch.cat([h[row], h[col], d2, edge_attr], dim=-1))
        cu  = scatter(rel * self.coord_mlp(m), row, dim=0,
                      dim_size=pos.size(0), reduce='add')
        cnt = scatter(torch.ones(edge_index.size(1), 1, device=pos.device),
                      row, dim=0, dim_size=pos.size(0), reduce='add')
        return out, cu / (cnt + 1e-8)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        out = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='add')
        cnt = scatter(torch.ones(inputs.size(0), 1, device=inputs.device),
                      index, dim=0, dim_size=dim_size, reduce='add')
        return out / (cnt + 1e-8)

    def update(self, aggr_out, h):
        return self.node_mlp(torch.cat([h, aggr_out], dim=-1))


class EGNN_NMR(nn.Module):
    """Single-target EGNN for NMR shielding. Works for both base and PFP inputs."""
    def __init__(self, input_dim, hidden_dim=128, edge_dim=1, num_layers=7):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim//4), nn.SiLU(),
            nn.Linear(hidden_dim//4, hidden_dim//4)
        )
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim//4) for _ in range(num_layers)
        ])
        self.metal_emb = nn.Embedding(N_METALS, METAL_EMB_DIM)
        head_in = hidden_dim + METAL_EMB_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch):
        h   = self.node_enc(batch.x)
        ef  = self.edge_enc(batch.edge_attr)
        pos = batch.pos
        for layer in self.layers:
            h, pos = layer(h, batch.edge_index, ef, pos)
        metal_h = h[batch.metal_mask]
        me = self.metal_emb(batch.metal_idx)
        z  = torch.cat([metal_h, me], dim=-1)
        return self.head(z).view(-1)


# ── Normalization helpers ──────────────────────────────────────────────────────
def load_stats():
    with open(f"{SPLITS_DIR}/nmr_shielding_stats.json") as f:
        return json.load(f)

def normalize(vals, mu, sigma):
    return (vals - mu) / (sigma + 1e-8)

def denormalize(vals, mu, sigma):
    return vals * sigma + mu


# ── Training ──────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, mu, sigma, field):
    model.train()
    tot_loss = tot_mae = 0.0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target_raw = getattr(batch, field).view(-1).to(device)
        target = normalize(target_raw, mu, sigma)
        loss = nn.functional.mse_loss(pred, target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tot_loss += loss.item()
        with torch.no_grad():
            tot_mae += (denormalize(pred.detach(), mu, sigma) - target_raw).abs().mean().item()
    n = len(loader)
    return tot_loss/n, tot_mae/n


@torch.no_grad()
def eval_epoch(model, loader, device, mu, sigma, field):
    model.eval()
    tot_mae = 0.0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target_raw = getattr(batch, field).view(-1).to(device)
        tot_mae += (denormalize(pred, mu, sigma) - target_raw).abs().mean().item()
    return tot_mae / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',   choices=['egnn', 'pfp'],              required=True)
    parser.add_argument('--target', choices=['rel', 'nonrel', 'delta'],   required=True)
    parser.add_argument('--gpu',    type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TARGET_MAP = {
        'rel':    ('y_nmr_rel',    'y_nmr_rel'),
        'nonrel': ('y_nmr_nonrel', 'y_nmr_nonrel'),
        'delta':  ('y_delta_nmr',  'y_delta_nmr'),
    }
    field, stats_key = TARGET_MAP[args.target]

    if args.arch == 'egnn':
        input_dim, edge_dim, graph_prefix = 5, 1, 'nmr_base'
    else:
        input_dim, edge_dim, graph_prefix = 261, 5, 'nmr_pfp'

    tag = f"{args.arch}_{args.target}"
    output_dir = f"/home/users/uchiyama/relativistic_effect/finetune_nmr_{tag}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}  arch={args.arch}  target={args.target}")

    stats = load_stats()
    mu    = stats[stats_key]['mean']
    sigma = stats[stats_key]['std']

    train_gs = torch.load(f"{SPLITS_DIR}/{graph_prefix}_train_graphs.pt", map_location='cpu', weights_only=False)
    val_gs   = torch.load(f"{SPLITS_DIR}/{graph_prefix}_val_graphs.pt",   map_location='cpu', weights_only=False)
    test_gs  = torch.load(f"{SPLITS_DIR}/{graph_prefix}_test_graphs.pt",  map_location='cpu', weights_only=False)
    print(f"Split: train={len(train_gs)}, val={len(val_gs)}, test={len(test_gs)}")

    train_loader = DataLoader(train_gs, batch_size=32, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_gs,   batch_size=64, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_gs,  batch_size=64, shuffle=False, num_workers=4)

    torch.manual_seed(42)
    model = EGNN_NMR(input_dim=input_dim, hidden_dim=128, edge_dim=edge_dim, num_layers=7).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    num_epochs = 500
    optimizer  = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-8)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    best_val_mae = float('inf')

    print(f"\n{'='*60}")
    print(f"EGNN({'no PFP' if args.arch=='egnn' else '+PFP'})  target={args.target}  {num_epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_mae = train_epoch(model, train_loader, optimizer, device, mu, sigma, field)
        vl_mae          = eval_epoch(model, val_loader, device, mu, sigma, field)
        scheduler.step()

        tag_star = ''
        if vl_mae < best_val_mae:
            best_val_mae = vl_mae
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_mae': vl_mae},
                       f"{output_dir}/best_model.pth")
            tag_star = ' ★'

        if epoch % 50 == 0 or epoch == 1:
            print(f"Ep {epoch:3d}/{num_epochs}  loss={tr_loss:.4f}  "
                  f"train={tr_mae:.1f}  val={vl_mae:.1f} ppm{tag_star}")

    ckpt = torch.load(f"{output_dir}/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    te_mae = eval_epoch(model, test_loader, device, mu, sigma, field)

    print(f"\n{'='*60}")
    print(f"Best epoch: {ckpt['epoch']}")
    print(f"Test MAE ({args.target}): {te_mae:.2f} ppm")

    result = {
        'model':      f"EGNN({'no PFP' if args.arch=='egnn' else '+PFP'})",
        'arch':       args.arch,
        'target':     args.target,
        'best_epoch': ckpt['epoch'],
        'n_train': len(train_gs), 'n_val': len(val_gs), 'n_test': len(test_gs),
        'test_mae_ppm': te_mae,
        'val_mae_ppm':  ckpt['val_mae'],
    }
    with open(f"{output_dir}/stats.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {output_dir}")


if __name__ == '__main__':
    main()

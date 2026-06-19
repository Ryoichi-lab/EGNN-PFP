#!/usr/bin/env python3
"""
Build train/val/test graph splits for NMR shielding prediction.
- 6,738 molecules with both ADF (ZORA) and ORCA (PBE0) complete
- Stratified split by metal element (8:1:1)
- PFP graphs: x=261-dim, edge_attr=5-dim
- Baseline graphs: x=5-dim (atomic features), edge_attr=1-dim (distance)
"""

import os, json, random
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from pathlib import Path

SEED     = 42
TRAIN_R  = 0.8
VAL_R    = 0.1
# TEST_R  = 0.1

BASE     = Path("/home/users/uchiyama/relativistic_effect")
NMR_CSV  = BASE / "finetune" / "nmr_shieldings.csv"
OUT_DIR  = BASE / "finetune"

# Load NMR shieldings
df = pd.read_csv(NMR_CSV)
print(f"NMR molecules: {len(df)}")
print(df['metal'].value_counts().to_string())

# Load ALL existing PFP graphs to build a csd_id -> graph lookup
# Use scalar_soc splits (which cover full B+C set)
print("\nLoading existing PFP graphs ...")
all_pfp = []
for split in ["train", "val", "test"]:
    gs = torch.load(OUT_DIR / f"scalar_soc_{split}_graphs.pt",
                    map_location="cpu", weights_only=False)
    all_pfp.extend(gs)
print(f"  PFP graphs loaded: {len(all_pfp)}")

pfp_lookup = {g.mol_id: g for g in all_pfp}

# Load ALL existing baseline graphs
print("Loading existing baseline graphs ...")
all_base = []
for split in ["train", "val", "test"]:
    fp = OUT_DIR / f"baseline_{split}_graphs.pt"
    if fp.exists():
        gs = torch.load(fp, map_location="cpu", weights_only=False)
        all_base.extend(gs)
print(f"  Baseline graphs loaded: {len(all_base)}")

base_lookup = {g.mol_id: g for g in all_base}

# ── Match NMR data to graphs ───────────────────────────────────────────────────
pfp_graphs  = []
base_graphs = []
missing     = 0

for _, row in df.iterrows():
    csd = row["csd_id"]
    s_rel    = float(row["shield_rel"])
    s_nonrel = float(row["shield_nonrel"])
    d_shield = float(row["delta_shield"])

    if csd not in pfp_lookup:
        missing += 1
        continue

    # PFP graph
    g = pfp_lookup[csd]
    import copy
    gp = copy.copy(g)
    n_atoms = gp.x.shape[0]
    metal_mask = torch.zeros(n_atoms, dtype=torch.bool)
    metal_mask[0] = True  # 金属は常にatom 0
    gp.metal_mask   = metal_mask
    gp.y_nmr_rel    = torch.tensor([s_rel],    dtype=torch.float32)
    gp.y_nmr_nonrel = torch.tensor([s_nonrel], dtype=torch.float32)
    gp.y_delta_nmr  = torch.tensor([d_shield], dtype=torch.float32)
    pfp_graphs.append(gp)

    # Baseline graph
    if csd in base_lookup:
        gb = copy.copy(base_lookup[csd])
        n_atoms_b = gb.x.shape[0]
        metal_mask_b = torch.zeros(n_atoms_b, dtype=torch.bool)
        metal_mask_b[0] = True
        gb.metal_mask   = metal_mask_b
        gb.y_nmr_rel    = torch.tensor([s_rel],    dtype=torch.float32)
        gb.y_nmr_nonrel = torch.tensor([s_nonrel], dtype=torch.float32)
        gb.y_delta_nmr  = torch.tensor([d_shield], dtype=torch.float32)
        base_graphs.append(gb)

print(f"\nMatched PFP graphs: {len(pfp_graphs)}, missing: {missing}")
print(f"Matched baseline graphs: {len(base_graphs)}")

# ── Stratified split by metal ──────────────────────────────────────────────────
def stratified_split(graphs, train_r, val_r, seed):
    rng = random.Random(seed)
    by_metal = defaultdict(list)
    for g in graphs:
        by_metal[g.metal].append(g)

    train, val, test = [], [], []
    for metal, gs in sorted(by_metal.items()):
        rng.shuffle(gs)
        n = len(gs)
        n_train = max(1, int(n * train_r))
        n_val   = max(1, int(n * val_r))
        train += gs[:n_train]
        val   += gs[n_train:n_train+n_val]
        test  += gs[n_train+n_val:]
        print(f"  {metal:3s}: total={n:4d}  train={len(gs[:n_train]):4d}  "
              f"val={len(gs[n_train:n_train+n_val]):3d}  "
              f"test={len(gs[n_train+n_val:]):3d}")
    return train, val, test

print("\n=== PFP split (stratified by metal) ===")
pfp_train, pfp_val, pfp_test = stratified_split(pfp_graphs, TRAIN_R, VAL_R, SEED)

print("\n=== Baseline split (stratified by metal) ===")
base_train, base_val, base_test = stratified_split(base_graphs, TRAIN_R, VAL_R, SEED)

# ── Compute normalization stats (train set only) ───────────────────────────────
def get_stats(graphs, field):
    vals = torch.cat([getattr(g, field) for g in graphs])
    return float(vals.mean()), float(vals.std())

stats = {}
for field in ["y_nmr_rel", "y_nmr_nonrel", "y_delta_nmr"]:
    mu, sigma = get_stats(pfp_train, field)
    stats[field] = {"mean": mu, "std": sigma}
    print(f"{field}: mean={mu:.2f}, std={sigma:.2f} ppm")

with open(OUT_DIR / "nmr_shielding_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# ── Save splits ────────────────────────────────────────────────────────────────
for tag, data in [("train", pfp_train), ("val", pfp_val), ("test", pfp_test)]:
    torch.save(data, OUT_DIR / f"nmr_pfp_{tag}_graphs.pt")
    print(f"Saved nmr_pfp_{tag}: {len(data)}")

for tag, data in [("train", base_train), ("val", base_val), ("test", base_test)]:
    torch.save(data, OUT_DIR / f"nmr_base_{tag}_graphs.pt")
    print(f"Saved nmr_base_{tag}: {len(data)}")

print("\nDone.")

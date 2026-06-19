#!/usr/bin/env python3
"""
Parse ADF (ZORA, relativistic) and ORCA (PBE0, non-relativistic) NMR output files.
Extracts isotropic metal NMR shielding (ppm) for each completed molecule.
Output: nmr_shieldings.csv  [csd_id, metal, mol_idx, shield_rel, shield_nonrel, delta_shield]
"""

import os, re, gzip, glob
from typing import Optional
import pandas as pd
from pathlib import Path

BASE = Path("/home/users/uchiyama/relativistic_effect")
ADF_DIRS  = [BASE / "input_nmr" / d for d in
             ["batch1_B_all","batch2_C_sm_med1","batch3_C_med2","batch4_C_lg1","batch5_C_lg2"]]
ORCA_DIRS = [BASE / "input_orca_nmr_nonrel" / d for d in
             ["batch_a","batch_b","batch_c","batch_d","batch_e"]]
OUT_CSV   = BASE / "finetune" / "nmr_shieldings.csv"

# ── CSD mapping ────────────────────────────────────────────────────────────────
b_map = pd.read_csv(BASE / "mol_idx_to_csd.csv")          # B set (2479)
c_map = pd.read_csv(BASE / "mol_idx_to_csd_optionC.csv")  # C set (7411)
all_map = pd.concat([b_map, c_map], ignore_index=True)
# key: (metal, mol_idx) -> csd_id
csd_lookup = {(r.metal, r.mol_idx): r.csd_id for _, r in all_map.iterrows()}

# ── Parsers ────────────────────────────────────────────────────────────────────
def parse_adf(path: str) -> Optional[float]:
    """Extract 'total isotropic shielding' from ADF ZORA NMR output."""
    try:
        with gzip.open(path, 'rt', errors='replace') as f:
            for line in f:
                if "total isotropic shielding" in line:
                    m = re.search(r'=\s*([-\d.]+)', line)
                    if m:
                        return float(m.group(1))
    except Exception:
        pass
    return None

def parse_orca(path: str) -> Optional[float]:
    """Extract metal (nucleus 0) isotropic shielding from ORCA output.
    Format:
        CHEMICAL SHIELDING SUMMARY (ppm)
        --------------------------------
        (blank lines / header)
          Nucleus  Element    Isotropic     Anisotropy
          -------  -------  ------------   ------------
              0       Ag         190.322        141.272
    """
    try:
        with gzip.open(path, 'rt', errors='replace') as f:
            in_summary = False
            skip_lines = 0
            for line in f:
                if "CHEMICAL SHIELDING SUMMARY" in line:
                    in_summary = True
                    skip_lines = 5  # skip: dashes + 2 blank + Nucleus header + ------- line
                    continue
                if in_summary:
                    if skip_lines > 0:
                        skip_lines -= 1
                        continue
                    # table row:  "      0       Ag         190.322        141.272"
                    m = re.match(r'\s+0\s+\w+\s+([-\d.]+)', line)
                    if m:
                        return float(m.group(1))
                    # end of table (blank line or new section)
                    if line.strip() == '' or re.match(r'\s*[A-Z]{3,}', line):
                        break
    except Exception:
        pass
    return None

# ── Collect ADF shieldings ─────────────────────────────────────────────────────
print("Parsing ADF outputs ...")
adf_shield = {}  # (metal, mol_idx) -> float
for d in ADF_DIRS:
    for fpath in sorted(d.glob("*.out.gz")):
        stem = fpath.stem.replace(".out", "")
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        metal, mid = parts[0], int(parts[1])
        val = parse_adf(str(fpath))
        if val is not None:
            adf_shield[(metal, mid)] = val

print(f"  ADF parsed: {len(adf_shield)}")

# ── Collect ORCA shieldings ────────────────────────────────────────────────────
print("Parsing ORCA outputs ...")
orca_shield = {}
for d in ORCA_DIRS:
    for fpath in sorted(d.glob("*.out.gz")):
        stem = fpath.stem.replace(".out", "")
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        metal, mid = parts[0], int(parts[1])
        val = parse_orca(str(fpath))
        if val is not None:
            orca_shield[(metal, mid)] = val

print(f"  ORCA parsed: {len(orca_shield)}")

# ── Build combined table ───────────────────────────────────────────────────────
rows = []
both_keys = set(adf_shield) & set(orca_shield)
print(f"Both complete: {len(both_keys)}")

for (metal, mol_idx) in sorted(both_keys):
    csd_id = csd_lookup.get((metal, mol_idx))
    if csd_id is None:
        continue
    s_rel    = adf_shield[(metal, mol_idx)]
    s_nonrel = orca_shield[(metal, mol_idx)]
    rows.append({
        "csd_id":       csd_id,
        "metal":        metal,
        "mol_idx":      mol_idx,
        "shield_rel":   s_rel,
        "shield_nonrel":s_nonrel,
        "delta_shield": s_rel - s_nonrel,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df)} molecules to {OUT_CSV}")
print(df.describe())

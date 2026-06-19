#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qm9_xyz_only フォルダ内の .xyz ファイルに原子数行を追加して修復
"""

from pathlib import Path

input_dir = Path("/home/users/uchiyama/QM9_dipole/qm9_xyz_only")
output_dir = Path("/home/users/uchiyama/QM9_dipole/qm9_xyz_fixed")
output_dir.mkdir(exist_ok=True)

xyz_files = sorted(input_dir.glob("*.xyz"))
print(f"Found {len(xyz_files)} files")

for xyz_file in xyz_files:
    with open(xyz_file, "r") as f:
        lines = [l for l in f.readlines() if l.strip()]
    natoms = len(lines)
    output_path = output_dir / xyz_file.name
    with open(output_path, "w") as f:
        f.write(f"{natoms}\n")
        f.write("Generated for ASE compatibility\n")
        f.writelines(lines)

print("✅ 修正完了: 原子数付きxyzを出力しました。")

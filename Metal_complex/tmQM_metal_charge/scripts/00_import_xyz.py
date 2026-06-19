#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gzip
import re

# ===== パス設定 =====
BASE_DIR = "/home/users/uchiyama/tmQM_dipole/datasets/tmqm/tmqm/tmQM"
XYZ_FILES = [os.path.join(BASE_DIR, f"tmQM_X{i}.xyz.gz") for i in (1, 2, 3)]
OUTPUT_DIR = os.path.join(BASE_DIR, "xyz_neutral_singlet")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 正規表現でヘッダー解析 =====
# 例: "CSD_code = WELROW | q = 0 | S = 0 | Stoichiometry = ... | MND = 8 | ..."
header_pat = re.compile(
    r"CSD_code\s*=\s*([A-Za-z0-9]+).*?"
    r"\bq\s*=\s*([+-]?\d+(?:\.\d+)?)\b.*?"
    r"\bS\s*=\s*([+-]?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)

tol = 1e-6
total_mols = 0
out_count = 0

def process_file(xyz_gz_path: str):
    global total_mols, out_count
    print(f"🔍 Processing {os.path.basename(xyz_gz_path)} ...")
    with gzip.open(xyz_gz_path, "rt") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            # 1行目: 原子数
            try:
                n_atoms = int(line)
            except ValueError:
                # ゴミ行等スキップ
                continue

            # 2行目: ヘッダー
            header = f.readline()
            if not header:
                break
            header = header.strip()

            # 原子座標 n_atoms 行
            atoms = []
            for _ in range(n_atoms):
                atom_line = f.readline()
                if not atom_line:
                    break
                atoms.append(atom_line)
            if len(atoms) != n_atoms:
                # 途中でEOF等なら終了
                break

            total_mols += 1
            m = header_pat.search(header)
            if not m:
                continue

            csd, q_str, s_str = m.groups()
            try:
                q = float(q_str)
                s = float(s_str)
            except ValueError:
                continue

            # 中性(q≈0) & 閉殻(S≈0)
            if abs(q) <= tol and abs(s) <= tol:
                out_path = os.path.join(OUTPUT_DIR, f"{csd}.xyz")
                with open(out_path, "w") as out:
                    out.write(f"{n_atoms}\n{header}\n")
                    out.writelines(atoms)
                out_count += 1
    print(f"✅ {os.path.basename(xyz_gz_path)} done")

def main():
    for p in XYZ_FILES:
        if not os.path.isfile(p):
            print(f"⚠️ 見つかりません: {p}")
            continue
        process_file(p)

    print("\n🎉 全処理完了！")
    print(f"総分子数: {total_mols:,}")
    print(f"条件一致 (q≈0, S≈0): {out_count:,} 分子を出力しました。")
    print(f"出力先: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

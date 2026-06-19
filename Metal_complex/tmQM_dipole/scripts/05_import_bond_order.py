#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tmQM 結合次数データの読み込み
"""

import gzip
import pickle
import numpy as np
from collections import defaultdict

def read_bond_orders(bo_file_path, debug=False):
    """
    結合次数ファイルを読み込む
    
    形式:
    CSD_code = WELROW | 2020-2024 CSD
         1  La  3.031        Se   4 0.429    Se   5 0.411    ...
         2  Se  1.967        P    8 1.313    La   1 0.285    ...
    
    Returns:
        dict: {mol_id: {(atom_i, atom_j): bond_order}}
    """
    bond_order_data = {}
    
    with gzip.open(bo_file_path, 'rt', encoding='utf-8', errors='ignore') as f:
        current_mol = None
        current_bonds = {}
        mol_count = 0
        
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # 空行をスキップ
            if not line:
                continue
            
            # CSD_code行
            if line.startswith('CSD_code'):
                # 前の分子を保存
                if current_mol and current_bonds:
                    bond_order_data[current_mol] = current_bonds
                    mol_count += 1
                    if debug and mol_count <= 3:
                        print(f"  Saved {current_mol}: {len(current_bonds)} bond pairs")
                
                # 新しい分子ID抽出
                # "CSD_code = WELROW | 2020-2024 CSD" → "WELROW"
                mol_id = line.split('=')[1].strip().split('|')[0].strip()
                current_mol = mol_id
                current_bonds = {}
                
                if debug and mol_count < 3:
                    print(f"\n  Processing {current_mol}...")
            
            # 原子行（数字で始まる）
            elif line and line[0].isdigit():
                parts = line.split()
                
                if len(parts) < 3:
                    continue
                
                try:
                    # 形式: "1  La  3.031  Se 4 0.429  Se 5 0.411 ..."
                    atom_idx = int(parts[0]) - 1  # 0-indexed
                    atom_type = parts[1]
                    atom_charge = float(parts[2])
                    
                    # parts[3]以降が結合情報: 原子タイプ、原子ID、結合次数の繰り返し
                    i = 3
                    bond_count = 0
                    while i < len(parts) - 1:
                        # 原子タイプ（アルファベット）
                        neighbor_type = parts[i]
                        i += 1
                        
                        if i >= len(parts):
                            break
                        
                        # 原子ID
                        try:
                            neighbor_idx = int(parts[i]) - 1  # 0-indexed
                            i += 1
                            
                            if i >= len(parts):
                                break
                            
                            # 結合次数
                            bond_order = float(parts[i])
                            i += 1
                            
                            # 双方向に格納
                            current_bonds[(atom_idx, neighbor_idx)] = bond_order
                            current_bonds[(neighbor_idx, atom_idx)] = bond_order
                            
                            bond_count += 1
                            
                            if debug and mol_count < 3 and bond_count <= 3:
                                print(f"    Bond: Atom {atom_idx+1}({atom_type}) - Atom {neighbor_idx+1}({neighbor_type}) = {bond_order:.3f}")
                        
                        except (ValueError, IndexError):
                            break
                
                except (ValueError, IndexError) as e:
                    if debug:
                        print(f"    Warning: Parse error on line {line_num}: {e}")
                    continue
        
        # 最後の分子を保存
        if current_mol and current_bonds:
            bond_order_data[current_mol] = current_bonds
            mol_count += 1
            if debug:
                print(f"  Saved {current_mol}: {len(current_bonds)} bond pairs")
    
    return bond_order_data


def analyze_bond_orders(bond_order_dict):
    """結合次数の統計を分析"""
    all_bond_orders = []
    
    for mol_id, bonds in bond_order_dict.items():
        for (i, j), bo in bonds.items():
            if i < j:  # 各結合を1回だけカウント
                all_bond_orders.append(bo)
    
    all_bond_orders = np.array(all_bond_orders)
    
    stats = {
        'total_bonds': len(all_bond_orders),
        'min': all_bond_orders.min(),
        'max': all_bond_orders.max(),
        'mean': all_bond_orders.mean(),
        'std': all_bond_orders.std(),
        'median': np.median(all_bond_orders),
        'percentiles': {
            '25%': np.percentile(all_bond_orders, 25),
            '50%': np.percentile(all_bond_orders, 50),
            '75%': np.percentile(all_bond_orders, 75),
            '90%': np.percentile(all_bond_orders, 90),
            '95%': np.percentile(all_bond_orders, 95)
        }
    }
    
    # 結合次数の分布
    single_bonds = np.sum((all_bond_orders > 0.8) & (all_bond_orders < 1.2))
    partial_double = np.sum((all_bond_orders >= 1.2) & (all_bond_orders < 1.8))
    double_bonds = np.sum((all_bond_orders >= 1.8) & (all_bond_orders < 2.5))
    triple_bonds = np.sum(all_bond_orders >= 2.5)
    weak_bonds = np.sum(all_bond_orders < 0.8)
    
    stats['distribution'] = {
        'weak (<0.8)': weak_bonds,
        'single (0.8-1.2)': single_bonds,
        'partial double (1.2-1.8)': partial_double,
        'double (1.8-2.5)': double_bonds,
        'triple (≥2.5)': triple_bonds
    }
    
    return stats, all_bond_orders


# メイン処理
if __name__ == "__main__":
    print("="*70)
    print("tmQM 結合次数データの読み込み")
    print("="*70)
    
    bond_order_dict = {}
    
    for i in range(1, 4):
        bo_file = f"/home/users/uchiyama/tmQM_dipole/datasets/tmqm/tmqm/tmQM/tmQM_X{i}.BO.gz"
        print(f"\nLoading {bo_file}...")
        
        # 最初のファイルのみデバッグ表示
        debug = (i == 1)
        bo_data = read_bond_orders(bo_file, debug=debug)
        
        bond_order_dict.update(bo_data)
        print(f"  ✓ Loaded {len(bo_data)} molecules")
    
    print(f"\n{'='*70}")
    print(f"Total: {len(bond_order_dict)} molecules with bond orders")
    
    # サンプル表示
    if bond_order_dict:
        print(f"\n{'='*70}")
        print("Sample Data")
        print("="*70)
        
        sample_mol = list(bond_order_dict.keys())[0]
        sample_bonds = bond_order_dict[sample_mol]
        
        print(f"\nMolecule: {sample_mol}")
        print(f"Number of bond pairs: {len(sample_bonds)}")
        print(f"\nSample bonds (first 10):")
        
        bond_list = [(i, j, bo) for (i, j), bo in sample_bonds.items() if i < j]
        bond_list.sort(key=lambda x: x[2], reverse=True)  # 結合次数でソート
        
        for i, j, bo in bond_list[:10]:
            print(f"  Atom {i+1:3d} - Atom {j+1:3d}: {bo:.3f}")
    
    # 統計情報
    print(f"\n{'='*70}")
    print("Bond Order Statistics")
    print("="*70)
    
    stats, all_bos = analyze_bond_orders(bond_order_dict)
    
    print(f"\nTotal bonds: {stats['total_bonds']:,}")
    print(f"Range: {stats['min']:.3f} - {stats['max']:.3f}")
    print(f"Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"Median: {stats['median']:.3f}")
    
    print(f"\nPercentiles:")
    for pct, value in stats['percentiles'].items():
        print(f"  {pct}: {value:.3f}")
    
    print(f"\nDistribution:")
    total = stats['total_bonds']
    for category, count in stats['distribution'].items():
        pct = count / total * 100
        print(f"  {category:25s}: {count:8d} ({pct:5.1f}%)")
    
    # 保存
    output_file = "/home/users/uchiyama/tmQM_dipole/bond_orders_tmqm.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(bond_order_dict, f)
    
    print(f"\n{'='*70}")
    print(f"✓ Saved to {output_file}")
    print("="*70)
    
    # 追加：分子名リストも保存
    mol_list_file = "/home/users/uchiyama/tmQM_dipole/bond_orders_mol_list.txt"
    with open(mol_list_file, 'w') as f:
        for mol_id in sorted(bond_order_dict.keys()):
            f.write(f"{mol_id}\n")
    
    print(f"✓ Molecule list saved to {mol_list_file}")
    print()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# MEC_v1ディレクトリをパスに追加（build_graph_utils.pyがある場所）
BASE_DIR = "/home/users/uchiyama/AEC_v6/MEC_v1"
sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from build_graph_utils import (
    build_node_features,
    build_edges,
)
from torch_geometric.data import Data
import re

# ===== 原子番号の辞書（全元素対応） =====
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
    'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
    'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
}

# ===== 絶対パス設定 =====
TMQM_BASE_DIR = "/home/users/uchiyama/tmQM_dipole"

# PFP記述子のパス（1から5まで）
PFP_DESCRIPTOR_PATHS = [
    os.path.join(TMQM_BASE_DIR, "PFP_descriptor", f"neutral_singlet_metal_{i}.npz")
    for i in range(1, 6)
]

# XYZファイルとdipole参照値のパス
XYZ_DIR = "/home/users/uchiyama/tmQM_dipole/datasets/tmqm/tmqm/tmQM/xyz_files"
DIPOLE_REFERENCE_PATH = "/home/users/uchiyama/tmQM_dipole/datasets/tmqm/tmqm/tmQM/tmQM_y.csv"

# 出力ディレクトリ（2種類）
OUTPUT_DIR_BASELINE = os.path.join(TMQM_BASE_DIR, "graphs_tmQM_baseline2")
OUTPUT_DIR_PFP = os.path.join(TMQM_BASE_DIR, "graphs_tmQM_pfp2")

os.makedirs(OUTPUT_DIR_BASELINE, exist_ok=True)
os.makedirs(OUTPUT_DIR_PFP, exist_ok=True)

print("="*60)
print("tmQMデータセット 金属錯体 双極子モーメント用グラフ作成")
print("EGNN-PFP 金属錯体拡張版")
print("="*60)
print("\n目的:")
print("  1. EGNNベースライン（PFPなし）")
print("  2. EGNN×PFP（PFP記述子あり）")
print("  の性能比較（金属錯体）")
print(f"\n作業ディレクトリ: {TMQM_BASE_DIR}")


# ===== XYZファイルから双極子モーメントを読み取る関数 =====
def read_xyz_with_dipole(xyz_path):
    """XYZファイルから原子座標と双極子モーメントを読み取る（tmQM形式対応）"""
    try:
        with open(xyz_path, 'r') as f:
            lines = f.readlines()
        
        # 1行目: 原子数
        n_atoms = int(lines[0].strip())
        
        # 2行目: コメント行（双極子情報を含む可能性）
        comment_line = lines[1].strip()
        
        # 双極子モーメント抽出（複数パターンに対応）
        dipole = None
        # パターン1: "dipole=1.234 D" または "dipole=1.234"
        match = re.search(r'dipole[=\s]+([\d.]+)', comment_line, re.IGNORECASE)
        if match:
            dipole = float(match.group(1))
        
        # 3行目以降: 原子座標
        atoms = []
        positions = []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            if len(parts) < 4:
                continue
            atoms.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        return atoms, np.array(positions), dipole
    
    except Exception as e:
        print(f"Error reading {xyz_path}: {e}")
        return None, None, None


def load_dipole_reference(reference_path):
    """
    tmQM_y.csvから双極子モーメントの参照値を読み込む
    
    フォーマット: ;Dispersion_E;Dipole_M;Metal
    XYZファイル名とCSD_codeを対応付ける
    """
    if reference_path is None:
        return {}
    
    try:
        # セミコロン区切りで読み込み
        df = pd.read_csv(reference_path, sep=';')
        
        print(f"  カラム: {df.columns.tolist()}")
        
        # Dipole_Mカラムが存在するか確認
        if 'Dipole_M' not in df.columns:
            print(f"⚠️ 警告: 'Dipole_M'カラムが見つかりません")
            return {}
        
        if 'CSD_code' not in df.columns:
            print(f"⚠️ 警告: 'CSD_code'カラムが見つかりません")
            return {}
        
        # CSD_code.xyz形式とCSD_codeの両方をキーとして登録
        dipole_dict = {}
        for idx, row in df.iterrows():
            csd_code = row['CSD_code']
            dipole = row['Dipole_M']
            
            # NaNチェック
            if pd.isna(csd_code) or pd.isna(dipole):
                continue
            
            # 両方の形式で登録
            dipole_dict[csd_code] = dipole
            dipole_dict[f"{csd_code}.xyz"] = dipole
        
        print(f"✓ {len(df)} 分子の双極子参照値を読み込みました")
        print(f"  双極子範囲: {df['Dipole_M'].min():.4f} - {df['Dipole_M'].max():.4f}")
        
        # サンプル表示（.xyz形式）
        sample_items = [(k, v) for k, v in list(dipole_dict.items())[:6] if k.endswith('.xyz')]
        print(f"  サンプル (.xyz形式): {sample_items[:3]}")
        
        return dipole_dict
        
    except Exception as e:
        print(f"⚠️ 双極子参照値の読み込みエラー: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
# ===== ベースライン用：PFPなしのグラフ作成関数（修正版） =====
def create_graph_from_xyz_baseline(xyz_path, mol_id, dipole_reference=None, cutoff=5.0, debug=False):
    """EGNNベースライン用：XYZファイルからグラフデータを作成（PFPなし）"""
    try:
        atoms, positions, dipole_from_xyz = read_xyz_with_dipole(xyz_path)
        
        if atoms is None or positions is None:
            if debug:
                print(f"  {mol_id}: 原子座標の読み込み失敗")
            return None
        
        # 双極子モーメントの取得優先順位
        dipole = None
        if dipole_reference and mol_id in dipole_reference:
            dipole = dipole_reference[mol_id]
        elif dipole_from_xyz is not None:
            dipole = dipole_from_xyz
        
        if dipole is None:
            if debug:
                print(f"  {mol_id}: 双極子モーメントが取得できません")
            return None
        
        # 原子番号に変換
        atomic_nums = np.array([ATOMIC_NUMBERS.get(atom, 1) for atom in atoms])
        
        # ダミーPFP記述子（ゼロベクトル）
        n_atoms = len(atoms)
        dummy_pfp = np.zeros((n_atoms, 256))
        
        # ノード特徴量の構築
        try:
            node_features = build_node_features(positions, atomic_nums, dummy_pfp)
            
            # PFP部分を除去（最初の256次元を削除）
            node_features = node_features[:, 256:]
            
            if debug:
                print(f"  {mol_id}: ノード特徴量 shape={node_features.shape}")
        except Exception as e:
            if debug:
                print(f"  {mol_id}: build_node_features エラー: {e}")
            raise
        
        # エッジの構築
        try:
            edge_index, edge_features = build_edges(positions, dummy_pfp, cutoff=cutoff)
            
            # エッジ特徴も簡略化（距離のみ）
            if len(edge_features) > 0:
                edge_features = [[ef[0]] for ef in edge_features]
            
            if debug:
                print(f"  {mol_id}: エッジ数={len(edge_index) if len(edge_index) > 0 else 0}")
        except Exception as e:
            if debug:
                print(f"  {mol_id}: build_edges エラー: {e}")
            raise
        
        return Data(
            x=torch.FloatTensor(node_features),
            pos=torch.FloatTensor(positions),
            edge_index=torch.LongTensor(edge_index).T if len(edge_index) > 0 else torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.FloatTensor(edge_features) if len(edge_features) > 0 else torch.zeros((0, 1)),
            y=torch.FloatTensor([dipole]),
            mol_id=mol_id
        )
    except Exception as e:
        if debug:
            print(f"  {mol_id}: グラフ作成エラー: {e}")
            import traceback
            traceback.print_exc()
        return None


# ===== PFP版：グラフ作成関数（修正版） =====
def create_graph_from_xyz_pfp(xyz_path, pfp_descriptors, mol_id, dipole_reference=None, cutoff=5.0, debug=False):
    """PFP版：XYZファイルとPFP記述子からグラフデータを作成"""
    try:
        atoms, positions, dipole_from_xyz = read_xyz_with_dipole(xyz_path)
        
        if atoms is None or positions is None:
            if debug:
                print(f"  {mol_id}: 原子座標の読み込み失敗")
            return None
        
        # 双極子モーメントの取得
        dipole = None
        if dipole_reference and mol_id in dipole_reference:
            dipole = dipole_reference[mol_id]
        elif dipole_from_xyz is not None:
            dipole = dipole_from_xyz
        
        if dipole is None:
            if debug:
                print(f"  {mol_id}: 双極子モーメントが取得できません")
            return None
        
        # 原子数の一致確認
        n_atoms = len(atoms)
        if pfp_descriptors.shape[0] != n_atoms:
            if debug:
                print(f"  {mol_id}: 原子数不一致 XYZ={n_atoms}, PFP={pfp_descriptors.shape[0]}")
            return None
        
        # 原子番号に変換
        atomic_nums = np.array([ATOMIC_NUMBERS.get(atom, 1) for atom in atoms])
        
        # ノード特徴量の構築
        try:
            node_features = build_node_features(positions, atomic_nums, pfp_descriptors)
            if debug:
                print(f"  {mol_id}: ノード特徴量 shape={node_features.shape}")
        except Exception as e:
            if debug:
                print(f"  {mol_id}: build_node_features エラー: {e}")
            raise
        
        # エッジの構築
        try:
            edge_index, edge_features = build_edges(positions, pfp_descriptors, cutoff=cutoff)
            if debug:
                print(f"  {mol_id}: エッジ数={len(edge_index) if len(edge_index) > 0 else 0}")
        except Exception as e:
            if debug:
                print(f"  {mol_id}: build_edges エラー: {e}")
            raise
        
        return Data(
            x=torch.FloatTensor(node_features),
            pos=torch.FloatTensor(positions),
            edge_index=torch.LongTensor(edge_index).T if len(edge_index) > 0 else torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.FloatTensor(edge_features) if len(edge_features) > 0 else torch.zeros((0, 4)),
            y=torch.FloatTensor([dipole]),
            mol_id=mol_id
        )
    except Exception as e:
        if debug:
            print(f"  {mol_id}: グラフ作成エラー: {e}")
            import traceback
            traceback.print_exc()
        return None

# ===== グラフデータ作成関数（ベースライン用） =====
def create_molecular_graphs_baseline(mol_ids, xyz_dir, dipole_reference, split_name):
    """ベースライン用：グラフデータ作成（PFPなし）"""
    graph_data_list = []
    failed_molecules = []
    
    print(f"\n{split_name}セットのグラフデータ作成中（ベースライン）...")
    for mol_id in tqdm(mol_ids):
        xyz_path = os.path.join(xyz_dir, mol_id)
        
        if not os.path.exists(xyz_path):
            failed_molecules.append(mol_id)
            continue
        
        graph_data = create_graph_from_xyz_baseline(
            xyz_path=xyz_path,
            mol_id=mol_id,
            dipole_reference=dipole_reference
        )
        
        if graph_data is not None:
            graph_data_list.append(graph_data)
        else:
            failed_molecules.append(mol_id)
    
    print(f"{split_name}: {len(graph_data_list)}個成功, {len(failed_molecules)}個失敗")
    if len(failed_molecules) > 0 and len(failed_molecules) <= 10:
        print(f"  失敗した分子例: {failed_molecules[:10]}")
    
    return graph_data_list


# ===== グラフデータ作成関数（PFP版） =====
def create_molecular_graphs_pfp(mol_ids, descriptor_data, xyz_dir, dipole_reference, split_name):
    """PFP版：グラフデータ作成"""
    graph_data_list = []
    failed_molecules = []
    
    print(f"\n{split_name}セットのグラフデータ作成中（PFP版）...")
    for mol_id in tqdm(mol_ids):
        xyz_path = os.path.join(xyz_dir, mol_id)
        
        if mol_id not in descriptor_data:
            failed_molecules.append(mol_id)
            continue
        
        pfp_descriptors = descriptor_data[mol_id]
        
        if not os.path.exists(xyz_path):
            failed_molecules.append(mol_id)
            continue
        
        graph_data = create_graph_from_xyz_pfp(
            xyz_path=xyz_path,
            pfp_descriptors=pfp_descriptors,
            mol_id=mol_id,
            dipole_reference=dipole_reference
        )
        
        if graph_data is not None:
            graph_data_list.append(graph_data)
        else:
            failed_molecules.append(mol_id)
    
    print(f"{split_name}: {len(graph_data_list)}個成功, {len(failed_molecules)}個失敗")
    if len(failed_molecules) > 0 and len(failed_molecules) <= 10:
        print(f"  失敗した分子例: {failed_molecules[:10]}")
    
    return graph_data_list

# ===== データ分割（金属種による層化分割） =====
def stratified_split_by_metal(mol_ids, xyz_dir, dipole_reference, train_ratio=0.8, val_ratio=0.1):
    """
    金属種ごとに層化してデータ分割
    
    Returns:
        train_mol_ids, val_mol_ids, test_mol_ids, metal_distribution
    """
    print("\n🔹 金属種を特定中...")
    
    # 金属元素リスト（tmQMで使用される遷移金属）
    METAL_ELEMENTS = {
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'
    }
    
    mol_metal_map = {}
    metal_counts = {}
    failed_read = []
    
    for mol_id in tqdm(mol_ids, desc="金属種特定"):
        xyz_path = os.path.join(xyz_dir, mol_id)
        
        if not os.path.exists(xyz_path):
            failed_read.append(mol_id)
            continue
        
        atoms, _, _ = read_xyz_with_dipole(xyz_path)
        
        if atoms is None:
            failed_read.append(mol_id)
            continue
        
        # 金属元素を抽出
        metals_in_mol = [atom for atom in atoms if atom in METAL_ELEMENTS]
        
        if len(metals_in_mol) == 0:
            mol_metal_map[mol_id] = 'no_metal'
        else:
            # 複数金属がある場合は最初の金属で分類（またはソートして結合）
            metal_key = metals_in_mol[0]
            mol_metal_map[mol_id] = metal_key
        
        # カウント
        metal_key = mol_metal_map[mol_id]
        metal_counts[metal_key] = metal_counts.get(metal_key, 0) + 1
    
    print(f"\n✓ 金属種の分布:")
    sorted_metals = sorted(metal_counts.items(), key=lambda x: x[1], reverse=True)
    for metal, count in sorted_metals[:15]:  # 上位15種表示
        print(f"  {metal}: {count} molecules ({count/len(mol_metal_map)*100:.1f}%)")
    
    if len(sorted_metals) > 15:
        print(f"  ... 他 {len(sorted_metals) - 15} 種類")
    
    # 層化分割
    print(f"\n🔹 金属種ごとに層化分割中...")
    train_ids = []
    val_ids = []
    test_ids = []
    
    test_ratio = 1.0 - train_ratio - val_ratio
    
    for metal, mol_list in tqdm(
        [(m, [mid for mid, mt in mol_metal_map.items() if mt == m]) 
         for m in set(mol_metal_map.values())],
        desc="層化分割"
    ):
        n_metal = len(mol_list)
        
        if n_metal == 0:
            continue
        
        # シャッフル（金属ごとに）
        np.random.shuffle(mol_list)
        
        # 最低1サンプルは各splitに（データが少ない金属種の場合）
        if n_metal < 3:
            train_ids.extend(mol_list)
            continue
        
        n_train = max(1, int(n_metal * train_ratio))
        n_val = max(1, int(n_metal * val_ratio))
        n_test = n_metal - n_train - n_val
        
        if n_test < 1:
            n_test = 1
            n_val = max(1, n_metal - n_train - n_test)
        
        train_ids.extend(mol_list[:n_train])
        val_ids.extend(mol_list[n_train:n_train + n_val])
        test_ids.extend(mol_list[n_train + n_val:])
    
    # 最終シャッフル
    np.random.shuffle(train_ids)
    np.random.shuffle(val_ids)
    np.random.shuffle(test_ids)
    
    # 分割後の金属分布を確認
    print(f"\n✓ 分割後の金属分布確認:")
    for split_name, split_ids in [("Train", train_ids), ("Val", val_ids), ("Test", test_ids)]:
        metal_dist = {}
        for mol_id in split_ids:
            metal = mol_metal_map.get(mol_id, 'unknown')
            metal_dist[metal] = metal_dist.get(metal, 0) + 1
        
        print(f"\n  {split_name} ({len(split_ids)} molecules):")
        sorted_dist = sorted(metal_dist.items(), key=lambda x: x[1], reverse=True)
        for metal, count in sorted_dist[:10]:
            print(f"    {metal}: {count} ({count/len(split_ids)*100:.1f}%)")
    
    return train_ids, val_ids, test_ids, metal_counts

# ===== メイン処理 =====
def main():
    # パスチェック
    if XYZ_DIR is None:
        print("❌ エラー: XYZ_DIRを設定してください")
        sys.exit(1)
    
    if not os.path.exists(XYZ_DIR):
        print(f"❌ エラー: XYZディレクトリが見つかりません: {XYZ_DIR}")
        sys.exit(1)
    
    # ===== PFP記述子ロード（1〜5を統合） =====
    print("\n🔹 tmQM PFP記述子をロード中...")
    descriptor_data = {}
    
    for i, pfp_path in enumerate(PFP_DESCRIPTOR_PATHS, 1):
        if not os.path.exists(pfp_path):
            print(f"⚠️  警告: PFP記述子ファイル{i}が見つかりません: {pfp_path}")
            continue
        
        data = np.load(pfp_path, allow_pickle=True)
        descriptor_data.update(data)
        print(f"✓ neutral_singlet_metal_{i}.npz: {len(data.keys())} molecules loaded")
    
    print(f"✓ 合計 {len(descriptor_data.keys())} molecules loaded")
    if len(descriptor_data) > 0:
        print(f"  サンプルキー: {list(descriptor_data.keys())[:3]}")
    
    # ===== 双極子参照値ロード =====
    print("\n🔹 双極子参照値をロード中...")
    dipole_reference = load_dipole_reference(DIPOLE_REFERENCE_PATH)
    if dipole_reference:
        print(f"✓ {len(dipole_reference)} エントリ（重複含む）を読み込みました")
    else:
        print("❌ エラー: 双極子参照値の読み込みに失敗しました")
        sys.exit(1)
    
    # ===== XYZファイルリスト取得 =====
    print(f"\n🔹 XYZファイルをスキャン中...")
    print(f"   パス: {XYZ_DIR}")
    
    xyz_files = sorted([f for f in os.listdir(XYZ_DIR) if f.endswith('.xyz')])
    print(f"✓ Total XYZ files: {len(xyz_files)}")
    print(f"  ファイル例: {xyz_files[:3]}")
    
    # ===== データの整合性チェック =====
    print("\n🔹 データの整合性チェック...")
    xyz_set = set(xyz_files)
    pfp_set = set(descriptor_data.keys())
    common_mols = xyz_set & pfp_set
    
    print(f"  XYZファイル数: {len(xyz_set)}")
    print(f"  PFP記述子の分子数: {len(pfp_set)}")
    print(f"  共通分子数: {len(common_mols)}")
    
    if len(common_mols) == 0:
        print(f"❌ エラー: XYZファイルとPFP記述子の間に共通の分子がありません")
        print(f"  XYZファイル例: {list(xyz_set)[:3]}")
        print(f"  PFPキー例: {list(pfp_set)[:3]}")
        sys.exit(1)
    
    # 共通分子のみを使用
    mol_ids = sorted(list(common_mols))
    print(f"✓ 使用する分子数: {len(mol_ids)}")
    
    # ===== サンプルデータ確認 =====
    print("\n🔹 サンプルデータの確認...")
    sample_xyz = os.path.join(XYZ_DIR, mol_ids[0])
    atoms, positions, dipole_xyz = read_xyz_with_dipole(sample_xyz)
    
    # 双極子参照値から取得
    dipole_ref = dipole_reference.get(mol_ids[0], None)
    
    if atoms is not None:
        print(f"  サンプル分子: {mol_ids[0]}")
        print(f"    原子数: {len(atoms)}")
        print(f"    双極子モーメント（XYZ）: {dipole_xyz if dipole_xyz else 'N/A'} Debye")
        print(f"    双極子モーメント（参照）: {dipole_ref if dipole_ref else 'N/A'} Debye")
        print(f"    原子種: {set(atoms)}")
        
        if dipole_ref is None and dipole_xyz is None:
            print(f"\n⚠️  警告: サンプル分子の双極子モーメントが取得できません")
            print(f"  デバッグ情報:")
            print(f"    - mol_id: {mol_ids[0]}")
            print(f"    - dipole_reference内: {mol_ids[0] in dipole_reference}")
            print(f"    - 参照値サンプル: {list(dipole_reference.items())[:3]}")
    else:
        print(f"❌ エラー: サンプルデータの読み込みに失敗しました")
        sys.exit(1)
    
    # ===== データ分割（層化分割） =====
    print("\n" + "="*60)
    print("🔹 データ分割（金属種による層化分割）")
    print("="*60)
    print("  目的: Train/Val/Testで金属種の分布を均等にする")
    print("  分割比率: Train 80% / Val 10% / Test 10%")
    
    n_total = len(mol_ids)
    
    # 層化分割実行
    train_mol_ids, val_mol_ids, test_mol_ids, metal_counts = stratified_split_by_metal(
        mol_ids, XYZ_DIR, dipole_reference, 
        train_ratio=0.8, val_ratio=0.1
    )
    
    print(f"\n✓ 層化分割完了:")
    print(f"  Train: {len(train_mol_ids)} molecules ({len(train_mol_ids)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_mol_ids)} molecules ({len(val_mol_ids)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_mol_ids)} molecules ({len(test_mol_ids)/n_total*100:.1f}%)")
    
    # 金属分布の統計をJSONに保存
    metal_distribution_stats = {
        'total_molecules': n_total,
        'unique_metals': len(metal_counts),
        'metal_counts': metal_counts,
        'stratified_split': True
    }

    
    # ===== グラフ生成（2種類） =====
    print("\n" + "="*60)
    print("🔹 グラフ生成開始")
    print("="*60)
    
    # 1. ベースライン（PFPなし）
    print("\n【1/2】EGNNベースライン（PFPなし）グラフ生成中...")
    train_graphs_baseline = create_molecular_graphs_baseline(
        train_mol_ids, XYZ_DIR, dipole_reference, "tmQM_Baseline_Train"
    )
    val_graphs_baseline = create_molecular_graphs_baseline(
        val_mol_ids, XYZ_DIR, dipole_reference, "tmQM_Baseline_Val"
    )
    test_graphs_baseline = create_molecular_graphs_baseline(
        test_mol_ids, XYZ_DIR, dipole_reference, "tmQM_Baseline_Test"
    )
    
    print(f"\n✓ ベースライングラフ作成完了:")
    print(f"  Train: {len(train_graphs_baseline)} graphs")
    print(f"  Val:   {len(val_graphs_baseline)} graphs")
    print(f"  Test:  {len(test_graphs_baseline)} graphs")
    
    # 2. PFP版
    print("\n【2/2】EGNN×PFP グラフ生成中...")
    train_graphs_pfp = create_molecular_graphs_pfp(
        train_mol_ids, descriptor_data, XYZ_DIR, dipole_reference, "tmQM_PFP_Train"
    )
    val_graphs_pfp = create_molecular_graphs_pfp(
        val_mol_ids, descriptor_data, XYZ_DIR, dipole_reference, "tmQM_PFP_Val"
    )
    test_graphs_pfp = create_molecular_graphs_pfp(
        test_mol_ids, descriptor_data, XYZ_DIR, dipole_reference, "tmQM_PFP_Test"
    )
    
    print(f"\n✓ PFPグラフ作成完了:")
    print(f"  Train: {len(train_graphs_pfp)} graphs")
    print(f"  Val:   {len(val_graphs_pfp)} graphs")
    print(f"  Test:  {len(test_graphs_pfp)} graphs")
    
    # ===== エラーチェック（詳細デバッグ） =====
    if len(train_graphs_baseline) == 0 and len(train_graphs_pfp) == 0:
        print("\n❌ エラー: グラフが1つも作成されませんでした")
        print("\n🔍 詳細デバッグを実行中...")
        
        # サンプル分子で詳細確認
        test_mol = mol_ids[0]
        test_xyz = os.path.join(XYZ_DIR, test_mol)
        print(f"\n  テスト分子: {test_mol}")
        print(f"  XYZファイル存在: {os.path.exists(test_xyz)}")
        print(f"  dipole_reference内: {test_mol in dipole_reference}")
        
        if test_mol in dipole_reference:
            print(f"  双極子値: {dipole_reference[test_mol]}")
        
        atoms, pos, dip = read_xyz_with_dipole(test_xyz)
        if atoms:
            print(f"  原子数: {len(atoms)}")
            print(f"  原子種: {set(atoms)}")
            print(f"  XYZから読み取った双極子: {dip}")
            print(f"  座標 shape: {pos.shape}")
            
            # build_node_features のテスト
# build_node_features のテスト
            print(f"\n  build_node_features のテスト...")
            try:
                atomic_nums = np.array([ATOMIC_NUMBERS.get(atom, 1) for atom in atoms])
                dummy_pfp = np.zeros((len(atoms), 256))
                node_feat = build_node_features(pos, atomic_nums, dummy_pfp)
                print(f"    ✓ 成功: shape={node_feat.shape}")
            except Exception as e:
                print(f"    ✗ 失敗: {e}")
                import traceback
                traceback.print_exc()
            
            # build_edges のテスト
            print(f"\n  build_edges のテスト...")
            try:
                edge_idx, edge_feat = build_edges(pos, dummy_pfp, cutoff=5.0)
                print(f"    ✓ 成功: エッジ数={len(edge_idx)}")
            except Exception as e:
                print(f"    ✗ 失敗: {e}")
                import traceback
                traceback.print_exc()
            
            # 実際にグラフ作成を試行（デバッグモード）
            print(f"\n  グラフ作成の詳細テスト（ベースライン）...")
            graph = create_graph_from_xyz_baseline(
                test_xyz, test_mol, dipole_reference, cutoff=5.0, debug=True
            )
            if graph:
                print(f"    ✓ グラフ作成成功")
            else:
                print(f"    ✗ グラフ作成失敗")
            
            # PFP版も試行
            if test_mol in descriptor_data:
                print(f"\n  グラフ作成の詳細テスト（PFP版）...")
                graph_pfp = create_graph_from_xyz_pfp(
                    test_xyz, descriptor_data[test_mol], test_mol, 
                    dipole_reference, cutoff=5.0, debug=True
                )
                if graph_pfp:
                    print(f"    ✓ グラフ作成成功")
                else:
                    print(f"    ✗ グラフ作成失敗")
        
        sys.exit(1)
        
        # サンプル分子で詳細確認
        test_mol = mol_ids[0]
        test_xyz = os.path.join(XYZ_DIR, test_mol)
        print(f"\n  テスト分子: {test_mol}")
        print(f"  XYZファイル存在: {os.path.exists(test_xyz)}")
        print(f"  dipole_reference内: {test_mol in dipole_reference}")
        
        if test_mol in dipole_reference:
            print(f"  双極子値: {dipole_reference[test_mol]}")
        
        atoms, pos, dip = read_xyz_with_dipole(test_xyz)
        if atoms:
            print(f"  原子数: {len(atoms)}")
            print(f"  XYZから読み取った双極子: {dip}")
        
        sys.exit(1)
    
    # ===== 保存 =====
    print("\n🔹 グラフデータを保存中...")
    
    # ベースライン保存
    torch.save(train_graphs_baseline, os.path.join(OUTPUT_DIR_BASELINE, "train_graphs.pt"))
    torch.save(val_graphs_baseline, os.path.join(OUTPUT_DIR_BASELINE, "val_graphs.pt"))
    torch.save(test_graphs_baseline, os.path.join(OUTPUT_DIR_BASELINE, "test_graphs.pt"))
    
    # PFP版保存
    torch.save(train_graphs_pfp, os.path.join(OUTPUT_DIR_PFP, "train_graphs.pt"))
    torch.save(val_graphs_pfp, os.path.join(OUTPUT_DIR_PFP, "val_graphs.pt"))
    torch.save(test_graphs_pfp, os.path.join(OUTPUT_DIR_PFP, "test_graphs.pt"))
    
    # ===== 双極子統計情報を収集 =====
    print("\n🔹 双極子統計情報を収集中...")
    
    all_dipoles_baseline = []
    all_dipoles_pfp = []
    
    for graph in train_graphs_baseline + val_graphs_baseline + test_graphs_baseline:
        all_dipoles_baseline.append(graph.y.item())
    
    for graph in train_graphs_pfp + val_graphs_pfp + test_graphs_pfp:
        all_dipoles_pfp.append(graph.y.item())
    
    all_dipoles_baseline = np.array(all_dipoles_baseline)
    all_dipoles_pfp = np.array(all_dipoles_pfp)
    
    # ===== 統計情報を保存（両方） =====
    print("\n🔹 統計情報を保存中...")
    
    # ベースライン統計
    stats_baseline = {
        'dataset': 'tmQM',
        'model_type': 'EGNN_baseline',
        'features': 'atom_type_only (金属を含む)',
        'target': 'dipole_moment',
        'n_molecules_total': n_total,
        'n_molecules_success': len(all_dipoles_baseline),
        'success_rate': f"{len(all_dipoles_baseline)/n_total*100:.1f}%",
        'split_config': {
            'train': n_train,
            'val': n_val,
            'test': n_test
        },
        'random_seed': 42,
        'n_train_graphs': len(train_graphs_baseline),
        'n_val_graphs': len(val_graphs_baseline),
        'n_test_graphs': len(test_graphs_baseline),
        'dipole_statistics': {
            'total': {
                'min': float(all_dipoles_baseline.min()),
                'max': float(all_dipoles_baseline.max()),
                'mean': float(all_dipoles_baseline.mean()),
                'std': float(all_dipoles_baseline.std())
            }
        }
    }
    
    # PFP統計
    stats_pfp = {
        'dataset': 'tmQM',
        'model_type': 'EGNN_with_PFP',
        'features': 'atom_type + PFP_descriptors (Matlantis)',
        'target': 'dipole_moment',
        'n_molecules_total': n_total,
        'n_molecules_success': len(all_dipoles_pfp),
        'success_rate': f"{len(all_dipoles_pfp)/n_total*100:.1f}%",
        'split_config': {
            'train': n_train,
            'val': n_val,
            'test': n_test
        },
        'random_seed': 42,
        'n_train_graphs': len(train_graphs_pfp),
        'n_val_graphs': len(val_graphs_pfp),
        'n_test_graphs': len(test_graphs_pfp),
        'dipole_statistics': {
            'total': {
                'min': float(all_dipoles_pfp.min()),
                'max': float(all_dipoles_pfp.max()),
                'mean': float(all_dipoles_pfp.mean()),
                'std': float(all_dipoles_pfp.std())
            }
        }
    }
    
    import json
    with open(os.path.join(OUTPUT_DIR_BASELINE, 'dataset_stats.json'), 'w') as f:
        json.dump(stats_baseline, f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR_PFP, 'dataset_stats.json'), 'w') as f:
        json.dump(stats_pfp, f, indent=2)
    
    # ===== 最終サマリー =====
    print("\n" + "="*60)
    print("✅ tmQM 金属錯体 グラフ作成完了!")
    print("="*60)
    
    print(f"\n【ベースライン（PFPなし）】")
    print(f"  Train: {len(train_graphs_baseline)} graphs")
    print(f"  Val:   {len(val_graphs_baseline)} graphs")
    print(f"  Test:  {len(test_graphs_baseline)} graphs")
    print(f"  保存先: {OUTPUT_DIR_BASELINE}")
    print(f"  双極子範囲: {stats_baseline['dipole_statistics']['total']['min']:.4f} - {stats_baseline['dipole_statistics']['total']['max']:.4f} Debye")
    print(f"  双極子平均: {stats_baseline['dipole_statistics']['total']['mean']:.4f} ± {stats_baseline['dipole_statistics']['total']['std']:.4f} Debye")
    
    print(f"\n【EGNN×PFP（金属錯体）】")
    print(f"  Train: {len(train_graphs_pfp)} graphs")
    print(f"  Val:   {len(val_graphs_pfp)} graphs")
    print(f"  Test:  {len(test_graphs_pfp)} graphs")
    print(f"  保存先: {OUTPUT_DIR_PFP}")
    print(f"  双極子範囲: {stats_pfp['dipole_statistics']['total']['min']:.4f} - {stats_pfp['dipole_statistics']['total']['max']:.4f} Debye")
    print(f"  双極子平均: {stats_pfp['dipole_statistics']['total']['mean']:.4f} ± {stats_pfp['dipole_statistics']['total']['std']:.4f} Debye")
    
    print(f"\n{'='*60}")
    print(f"🎉 完了 - 2種類のグラフデータセット（金属錯体）を作成しました")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
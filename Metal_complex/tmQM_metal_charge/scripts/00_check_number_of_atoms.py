import numpy as np
import shutil
from pathlib import Path

def extract_large_molecules(pfp_dir, xyz_source_dir, output_dir):
    """256原子超の分子のXYZファイルを抽出"""
    pfp_dir = Path(pfp_dir)
    xyz_source_dir = Path(xyz_source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # ファイル別に分子をグループ化
    molecules_by_file = {i: [] for i in range(1, 6)}
    
    print("=== 256原子超の分子を検出中 ===\n")
    
    for i in range(1, 6):
        npz_file = pfp_dir / f"neutral_singlet_metal_{i}.npz"
        if not npz_file.exists():
            continue
        
        data = np.load(npz_file, allow_pickle=True)
        
        for xyz_name in data.keys():
            descriptors = data[xyz_name]
            n_atoms = descriptors.shape[0]
            
            if n_atoms > 256:
                molecules_by_file[i].append((xyz_name, n_atoms))
        
        data.close()
    
    # XYZファイルをコピー
    print("=== XYZファイルを抽出中 ===\n")
    
    total_extracted = 0
    extraction_info = []
    
    for file_num in range(1, 6):
        if not molecules_by_file[file_num]:
            continue
        
        print(f"File {file_num}: {len(molecules_by_file[file_num])} molecules")
        
        for xyz_name, n_atoms in molecules_by_file[file_num]:
            source_xyz = xyz_source_dir / xyz_name
            dest_xyz = output_dir / xyz_name
            
            if source_xyz.exists():
                shutil.copy2(source_xyz, dest_xyz)
                extraction_info.append({
                    'xyz_name': xyz_name,
                    'n_atoms': n_atoms,
                    'file_num': file_num
                })
                total_extracted += 1
                print(f"  ✓ {xyz_name} ({n_atoms} atoms)")
            else:
                print(f"  ✗ {xyz_name} NOT FOUND in source directory")
    
    # メタデータを保存
    metadata_file = output_dir / "large_molecules_metadata.npz"
    np.savez(metadata_file,
             xyz_names=[info['xyz_name'] for info in extraction_info],
             n_atoms=[info['n_atoms'] for info in extraction_info],
             file_nums=[info['file_num'] for info in extraction_info])
    
    print(f"\n=== 完了 ===")
    print(f"Total extracted: {total_extracted} XYZ files")
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved: {metadata_file}")
    
    return extraction_info

if __name__ == "__main__":
    pfp_dir = "/home/users/uchiyama/tmQM_dipole/PFP_descriptor"
    xyz_source_dir = "/home/users/uchiyama/tmQM_dipole/datasets/tmqm/tmqm/tmQM/xyz_neutral_singlet"
    output_dir = "/home/users/uchiyama/tmQM_dipole/large_molecules_for_recompute"
    
    extraction_info = extract_large_molecules(pfp_dir, xyz_source_dir, output_dir)
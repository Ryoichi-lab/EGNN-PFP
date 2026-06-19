#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QM9 U Baseline - プロット再作成スクリプト
保存済みのチェックポイントからプロットのみ作成
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.metrics import r2_score

# パス設定
CHECKPOINT_PATH = "/home/users/uchiyama/QM9_U/QM9_u_training_pfp_egnn_128_attention_perfect_0110/checkpoint_epoch_1000.pth"
OUTPUT_DIR = "/home/users/uchiyama/QM9_U/training_baseline_U_egnn_1216"

print("="*70)
print("QM9 U Baseline - プロット再作成")
print("="*70)

# チェックポイント読み込み
print(f"\n🔹 チェックポイント読み込み中...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

# 履歴データ取得
history = checkpoint['history']
best_val_mae = checkpoint['val_mae']
epoch = checkpoint['epoch']

print(f"✓ Epoch: {epoch}")
print(f"✓ Best Val MAE: {best_val_mae:.2f} meV")

# プロット作成
print(f"\n🔹 プロット作成中...")

PAPER_MAE_MEV = 43.0  # Paper EGNN MAE (U)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# MAE
ax1 = axes[0, 0]
ax1.plot(history['train_maes'], label='Train MAE', linewidth=2)
ax1.plot(history['val_maes'], label='Val MAE', linewidth=2)
ax1.axhline(y=PAPER_MAE_MEV, color='red', linestyle='--', linewidth=2, 
            label=f'Paper EGNN ({PAPER_MAE_MEV:.2f} meV)')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('MAE (meV)', fontsize=12, fontweight='bold')
ax1.set_title('Baseline EGNN (Official) - MAE (U)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Loss
ax2 = axes[0, 1]
ax2.plot(history['train_losses'], label='Train Loss (Normalized)', linewidth=2, color='orange')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('Training Loss (Normalized Scale)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Learning Rate
ax3 = axes[1, 0]
ax3.plot(history['learning_rates'], linewidth=2, color='green')
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
ax3.set_title('LR Schedule (Cosine)', fontsize=14, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# 散布図用のダミープロット（実際のval_predsがないため）
ax4 = axes[1, 1]
ax4.text(0.5, 0.5, f'Best Val MAE: {best_val_mae:.2f} meV\nEpoch: {epoch}', 
         ha='center', va='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax4.set_xlabel('True U (meV)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Predicted U (meV)', fontsize=12, fontweight='bold')
ax4.set_title('Baseline Official Results', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'training_history_baseline_official_replot.png')
plt.savefig(output_path, dpi=200)
plt.close()

print(f"✓ プロット保存: {output_path}")

# 統計情報を更新
print(f"\n🔹 統計情報更新中...")
stats = {
    'dataset': 'QM9',
    'functional': 'B3LYP/6-31G(2df,p)',
    'target': 'U (Internal energy at 298.15K)',
    'model_version': 'baseline_egnn_official_implementation',
    'official_implementation': True,
    'reference': 'EGNN (Satorras et al., 2021) - Official implementation',
    'training': {
        'best_val_mae_meV': float(best_val_mae),
        'paper_egnn_mae_meV': PAPER_MAE_MEV,
        'unit': 'meV',
        'epochs_trained': epoch,
        'batch_size': 32,
        'initial_lr': 1e-4,
        'weight_decay': 1e-8,
        'grad_clip': 1.0,
        'target_normalization': {
            'enabled': True,
            'mean_Ha': checkpoint.get('mean', 0.0),
            'mad_Ha': checkpoint.get('mad', 0.0)
        }
    },
    'model_config': {
        'input_features': '5-dim one-hot (H,C,N,O,F)',
        'edge_features': 'None',
        'hidden_dim': 128,
        'num_layers': 7,
        'coord_update': False,
        'coords_weight': 1.0,
        'residual': True,
        'attention': False,
        'node_attr': True,
        'pfp_enabled': False,
        'target_normalized': True
    }
}

stats_path = os.path.join(OUTPUT_DIR, 'training_stats_baseline_official_updated.json')
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"✓ 統計情報保存: {stats_path}")

print(f"\n{'='*70}")
print("🎉 プロット再作成完了！")
print(f"{'='*70}")
print(f"Best Val MAE: {best_val_mae:.2f} meV")
print(f"Paper EGNN MAE (U): {PAPER_MAE_MEV:.2f} meV")

improvement_ratio = (PAPER_MAE_MEV - best_val_mae) / PAPER_MAE_MEV * 100
if improvement_ratio > 0:
    print(f"Improvement over paper: {improvement_ratio:.1f}%")
else:
    print(f"Difference from paper: {abs(improvement_ratio):.1f}%")
print(f"{'='*70}\n")
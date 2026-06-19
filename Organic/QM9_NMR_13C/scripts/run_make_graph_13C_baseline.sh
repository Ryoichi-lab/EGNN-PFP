#!/bin/bash
#PBS -N make_graph_13C_base
#PBS -q workq
#PBS -l select=1:ncpus=8:mem=50GB:vnode=asteion01
#PBS -l walltime=1000:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
source ~/miniconda3/etc/profile.d/conda.sh
conda activate chemonto2

echo "Job started at $(date)"
python /home/users/uchiyama/qm9nmr/EGNN_PFP/make_graph_13C_baseline.py
echo "Job completed at $(date)"

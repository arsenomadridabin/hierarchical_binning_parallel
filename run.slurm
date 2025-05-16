#!/bin/bash
#SBATCH -N 1                    # number of nodes
#SBATCH -n 20                   # number of MPI processes
##SBATCH -c 12                  # 6 threads per MPI process
#SBATCH -t 15:00:00
#SBATCH -p checkpt
#SBATCH -A hpc_bb_karki3
#SBATCH -o  gb.out
#SBATCH -e  err.out

source /scratch/ashaky3/.env/bin/activate

python run.py \
  --fe_file fe.json \
  --mg_file mg.json \
  --si_file si.json \
  --o_file o.json \
  --n_file n.json \
  --cell_size 68.0 \
  --num_bins 8 \
  --sub_bins 2 \
  --partition_threshold 15 \
  --fe_rich_threshold 7 \
  --fe_poor_threshold 1 \
  --skip 20 \
  --save_intermediates

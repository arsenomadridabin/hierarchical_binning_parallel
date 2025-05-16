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

python parallel_v2.py   --fe_file fe.json   --mg_file mg.json   --si_file si.json   --o_file o.json   --n_file n.json   --partition_threshold 15   --fe_rich_threshold 5   --fe_poor_threshold 1 --skip 10 --nproc 20 --plot_k 5

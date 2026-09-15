#!/bin/bash
#SBATCH -J coconet_cpu
#SBATCH -A gts-phentenryck3
#SBATCH -q embers
#SBATCH -N 1 --ntasks-per-node=4
#SBATCH --mem-per-cpu=8G
#SBATCH -t 720
#SBATCH -o logs/%x_%j.out
# One CPU method on one dataset:  sbatch slurm/run_cpu.sbatch <dataset> <method> [run.py args...]
# The paper's CPU numbers were produced with 4 threads and no GPU visible.
cd "$(dirname "$0")/.." || exit 1
source ~/.bashrc; conda activate cp4ts
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
DATASET=${1:?dataset}; METHOD=${2:?method}; shift 2
python -W ignore run.py --dataset "$DATASET" --method "$METHOD" --config configs/paper.yaml "$@"

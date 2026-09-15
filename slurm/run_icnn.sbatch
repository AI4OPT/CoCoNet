#!/bin/bash
#SBATCH -J coconet_icnn
#SBATCH -A gts-phentenryck3
#SBATCH -q embers
#SBATCH -p gpu-rtx6000
#SBATCH -C RTX6000
#SBATCH -N 1 --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH -t 720
#SBATCH -o logs/%x_%j.out
# CoCoNet_ICNN on one dataset:  sbatch slurm/run_icnn.sbatch <dataset> [run.py args...]
# The paper's ICNN numbers were produced on a Quadro RTX 6000 (training and test volume).
cd "$(dirname "$0")/.." || exit 1
source ~/.bashrc; conda activate cp4ts
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATASET=${1:?dataset}; shift 1
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
python -W ignore run.py --dataset "$DATASET" --method icnn --config configs/paper.yaml "$@"

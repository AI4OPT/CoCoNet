#!/bin/bash
#SBATCH -J learnscore-tau1-tests
#SBATCH -A gts-phentenryck3
#SBATCH -q embers
#SBATCH -N 1 --ntasks-per-node=4
#SBATCH --mem-per-cpu=8G
#SBATCH -t 01:30:00
#SBATCH -o scripts/logs/tests_%j.out
#SBATCH -e scripts/logs/tests_%j.err

cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs

PY=/storage/home/hcoda1/1/amoradi30/.conda/envs/cp4ts/bin/python3
"$PY" -m pytest tests/ -v

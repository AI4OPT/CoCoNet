#!/bin/bash
# Submit every (dataset, method) job of the paper's tables: one job per pair, all five seeds.
#   bash slurm/submit_all.sh            # the 25 multivariate + 6 univariate datasets
#   bash slurm/submit_all.sh enb scpf   # a subset
cd "$(dirname "$0")/.." || exit 1
DATASETS=${@:-$(python -c "import yaml; print(' '.join(yaml.safe_load(open('configs/datasets.yaml'))))")}
for D in $DATASETS; do
  d=$(python -c "import yaml; print(yaml.safe_load(open('configs/datasets.yaml'))['$D']['output_dim'])")
  if [ "$d" = 1 ]; then METHODS="absolute_residual locally_weighted mvcs spline"; else METHODS="absolute_residual empirical_copula global_ellipsoid local_ellipsoid mvcs spline"; fi
  for M in $METHODS; do sbatch -J "${M:0:6}_$D" slurm/run_cpu.sbatch "$D" "$M"; done
  sbatch -J "icnn_$D" slurm/run_icnn.sbatch "$D"
done

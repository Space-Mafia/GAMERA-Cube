#!/usr/bin/env bash
#SBATCH --job-name=gamera-cube-earth
#SBATCH --nodes=2
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=2
#SBATCH --time=16:00:00
#SBATCH --output=earth-%j.log
set -euo pipefail
# Submit from the repository root. Add your site's account/partition options
# to sbatch, and load the SAME compiler/MPI/HDF5 modules used for compilation.
# Memory and MPI plugin selection are site-specific; consult your cluster.
python3 scripts/run_earth.py --profile reference --launcher srun \
  --output "runs/earth-reference-${SLURM_JOB_ID}"

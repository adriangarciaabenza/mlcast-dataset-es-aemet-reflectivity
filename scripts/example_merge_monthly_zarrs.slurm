#!/bin/bash
#SBATCH --job-name=merge_zarr_v3_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/merge_zarr_%j.out
#SBATCH --error=logs/merge_zarr_%j.err

set -euo pipefail

# This is an example SLURM script for merging monthly Zarr stores.
# Adapt paths, environment activation, and requested resources to your HPC system.

source /etc/profile

# -------------------------------------------------------------------------
# User configuration
# -------------------------------------------------------------------------
PROJECT_DIR="/path/to/mlcast-dataset-ES-AEMET-reflectivity"
VENV_DIR="${PROJECT_DIR}/.venv"

export PYTHONUNBUFFERED=1

# Limit thread oversubscription on HPC systems
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Optional Dask-related setting
export DASK_NUM_WORKERS="${SLURM_CPUS_PER_TASK}"

source "${VENV_DIR}/bin/activate"

echo "=== JOB START ==="
date
hostname
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE:-unknown}"
echo "Working directory: ${PROJECT_DIR}"

cd "${PROJECT_DIR}"

python scripts/merge_monthly_zarrs.py

echo "=== JOB END ==="
date